#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <sys/resource.h>
#include <time.h> 
#include <algorithm>
#include <chrono>
#include <random> 
#include <omp.h> 

using namespace std;

struct Problem
{
    Problem() {}

    int nr_instance, nr_field;

    std::map<std::string, int> feat2idmap;
    std::map<int, std::string> id2featmap;

    std::map<int, std::string> instidmap;
    std::vector<map<int, double>> X;

    std::vector<vector<pair<int, double>>> Xhat;

    std::vector<double> Y;
    std::vector<double> F;
    std::vector<double> B;
    std::vector<double> W;
};
struct Option
{
    Option() : nr_iter(100), nr_lr(0.002), nr_reg(0), feat_threshold(0.001) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter;
    double nr_lr, nr_reg, feat_threshold;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: Tobit [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-r <nr_reg>: set the reg\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    int const argc = static_cast<int>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    int i = 0;
    for(; i < argc; ++i)
    {
	if(args[i].compare("-i") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_iter = stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_lr = stof(args[++i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_reg = stof(args[++i]);
        }
        else
        {
            break;
        }
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

double pnorm(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

 // Save the sign of x
    int sign = 1;
    if (x < 0)
    	sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

double dnorm(double x,double miu, double sigma)
{
    return 1.0/(sqrt(2*M_PI)*sigma) * exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma)); 
}

bool mysortfunc(const pair<string, double>& a, const pair<string, double>& b)
{
    return a.second > b.second;
}

double sign(double w){
	if(w > opt.feat_threshold) return 1;
	else if(w < -opt.feat_threshold)return -1;
	else return 0;
}

void writeWeightFile(Problem& Tr)
{
	ofstream outfile("feature_weight_tobit");
	for (int f = 0; f < Tr.nr_field; f += 1)
	{
		string fname = Tr.id2featmap[f];
		outfile << fname << " " << Tr.W[f] << endl;
        }
	outfile.close();
}

void writeResultFile(Problem& Va) {
	ofstream outfile(opt.Va_out_path);
	for(int f = 0; f < Va.nr_instance; f+= 1) {
		outfile << Va.Y[f] << "\t" << Va.B[f] << "\t" << Va.F[f] << endl;
	}
	outfile.close();
}


double calAUC(Problem& prob)
{
	int poscnt = 0;
	vector<pair<string, double>> userid2score(prob.nr_instance);
	map<string, double> userid2label;
	for (int i = 0; i < prob.nr_instance; i += 1)
	{
		string username = prob.instidmap[i];
		userid2score[i] = pair<string, double>(username, prob.F[i]);

		int yi = prob.Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[username] = yi;
	}
	int negcnt = prob.nr_instance - poscnt;
	sort(userid2score.begin(), userid2score.end(), mysortfunc);

        ofstream outfile(opt.Va_out_path);

	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < userid2score.size(); ++i) 
	{
        	string username = userid2score[i].first;
                double fscore = userid2score[i].second;
		int yi = userid2label[username];
                outfile << username << "\t" << fscore << "\t" << yi << endl;

		if (yi == 1) 
		{
			score += belownegcnt * 1.0 / negcnt;
			predictposcnt += 1;
		} 
		else {
			belownegcnt -= 1;
		}

		if (i == startpoint) 
		{
			pvcnt.push_back(i);
			covcnt.push_back(predictposcnt);
			startpoint *= 2;
		}
	}
        outfile.close();
	pvcnt.push_back(userid2score.size());
	covcnt.push_back(predictposcnt);
	
	//for (int i = 0; i < pvcnt.size(); i += 1) cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	return auc;
}

void loadTrainInstance(Problem& Tr, ifstream& inputfile) 
{    
    string line;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
	double bidprice;
        int target;
        iss >> userid >> target >> bidprice;
	if (target != 1) target = 0;

	Tr.Y.push_back(target);
	Tr.B.push_back(bidprice);
	Tr.F.push_back(0);
	Tr.X.push_back(std::map<int, double>());

	int i = Tr.X.size() - 1;
	Tr.instidmap[i] = userid;

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		int f = -1;
		if (Tr.feat2idmap.count(feat) == 0) {
			// new feature
			int fid = Tr.feat2idmap.size();
			Tr.W.push_back(0);
			Tr.feat2idmap[feat] = fid;
			Tr.id2featmap[fid] = feat;
		}
		f = Tr.feat2idmap[feat];
		if (Tr.X[i].count(f) != 0) continue;
		Tr.X[i][f] = fval;
        }
    }
    Tr.nr_instance = Tr.X.size();
    Tr.nr_field = Tr.feat2idmap.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{    
    string line;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
	double bidprice;
        int target;
        iss >> userid >> target >> bidprice;
	if (target != 1) target = 0;

	Va.Y.push_back(target);
	Va.B.push_back(bidprice);
	Va.F.push_back(0);
	Va.X.push_back(std::map<int, double>());

	int i = Va.X.size() - 1;
	Va.instidmap[i] = userid;

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		if (Tr.feat2idmap.count(feat) == 0) continue;
		int j = Tr.feat2idmap[feat];
		Va.X[i][j] = fval;
        }
    }
    Va.nr_instance = Va.X.size();
}

std::vector<std::string> argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}


double getStdev(vector<double> resultSet) {
        double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
        double mean =  sum / resultSet.size(); //均值  

        double accum  = 0.0;
        std::for_each (std::begin(resultSet), std::end(resultSet), [&](const double d) {
                accum  += (d-mean)*(d-mean);
        });

	
        double stdev = sqrt(accum/(resultSet.size()-1)); //方差  

        return stdev;
}

double getSigma(Problem& Tr) {
	vector<double> observed(Tr.nr_instance);
	for (int i = 0; i < Tr.nr_instance; i += 1)
	{
		int yi = Tr.Y[i];
		double bidPrice = Tr.B[i];
		if(yi == 1)	observed.push_back(bidPrice);
	}
	return getStdev(observed);
}

int main(int const argc, char const * const * const argv) 
{
	try {
		opt = parse_option(argv_to_args(argc, argv));
	}
	catch(std::invalid_argument const &e) {	
		std::cout << e.what();
		return EXIT_FAILURE;
	}


	omp_set_num_threads(static_cast<int>(50));

	Problem Tr, Va;

	ifstream trainfile(opt.Tr_path);
	loadTrainInstance(Tr, trainfile);
	cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field <<  endl;

	ifstream testfile(opt.Va_path);
	loadTestInstance(Tr, Va, testfile);
	cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;

	Tr.Xhat.resize(Tr.nr_field, vector<pair<int, double>>());
        for (int i = 0; i < Tr.nr_instance; i += 1)
        {
                for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
                {
                        int f = it->first;
                        double x = it->second;
                        Tr.Xhat[f].push_back(pair<int, double>(i, x));
                }
        }

        for (int f = 0; f < Tr.nr_field; f += 1) Tr.Xhat[f].shrink_to_fit();
//	double sum = std::accumulate(std::begin(trainpricevec), std::end(resultSet), 0.0);
	double sigma = getSigma(Tr);
	cout << "iter\ttr_loss\tva_loss\ttr_mse\tva_mse\t" << sigma << endl;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		vector<double> trFVec(Tr.nr_instance,0);
		#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
			{
				int f = it->first;
			        double x = it->second;
			        trFVec[i] += Tr.W[f] * x;
			}
		}

		vector<double> gradientVec(Tr.nr_field, 0);
		#pragma omp parallel for schedule(dynamic)
		for (int f = 0; f < Tr.nr_field; f += 1)
		{
			for (int j = 0; j < Tr.Xhat[f].size(); j += 1)
			{
				pair<int, double>& ins = Tr.Xhat[f][j];
				int i = ins.first;
				double x = ins.second;
				double z = (trFVec[i] - Tr.B[i])/sigma;
				if(z < -3) z = -3;
				if(z > 3) z = 3;
				double dpnorm = -exp(log(dnorm(z,0,1)) - log(pnorm(z)));
				gradientVec[f] += (Tr.Y[i] * z * x / sigma) +(1 - Tr.Y[i]) * dpnorm * x / sigma;
			}
		}

		int wnum = 0;	
		#pragma omp parallel for schedule(static) reduction(+: wnum)
		for (int i = 0;i < Tr.W.size(); i += 1)
		{
			Tr.W[i] += - opt.nr_lr * gradientVec[i] + opt.nr_reg * Tr.W[i];
			if (fabs(Tr.W[i]) < opt.feat_threshold)
			{
				Tr.W[i] = 0;
				wnum += 1;
			}
		}

		double trainmse = 0.0;
		double trainloss = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: trainmse, trainloss)
        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
            		int yi = Tr.Y[i];
			double bidprice = Tr.B[i];
			double Fi = 0.0;
			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Tr.F[i] = Fi;
			double z = (Fi - bidprice)/sigma;
			if(z < -3) z = -3;
                        if(z > 3) z = 3;
			trainloss += - (yi * log(dnorm(z,0,1)) + (1 - yi) * log(pnorm(z)));
			trainmse += (bidprice - Tr.F[i]) * (bidprice - Tr.F[i]);
		}
		trainloss /= Tr.nr_instance;
		trainmse /= Tr.nr_instance;
		//double trainauc = calAUC(Tr);

		writeWeightFile(Tr);

		double testmse = 0;
		double testloss = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: testmse, testloss)
        	for (int i = 0; i < Va.nr_instance; i += 1) 
		{
            		int yi = Va.Y[i];
			double bidprice = Va.B[i];
			double Fi = 0.0;
			for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Va.F[i] = Fi;
			double z = (Fi - bidprice)/sigma;
			if(z < -3) z = -3;
                        if(z > 3) z = 3;
			testloss += - (yi * log(dnorm(z,0,1)) + (1 - yi) * log(pnorm(z)));
			testmse += (bidprice - Va.F[i]) * (bidprice - Va.F[i]);
		}
		testloss /= Va.nr_instance;
		testmse /= Va.nr_instance;

		writeResultFile(Va);
		//double testauc = calAUC(Va);
		cout << iter << "\t" << trainloss << "\t" << testloss << "\t" << "\t" << trainmse << "\t" << testmse << endl;
	}

return 0;
}
