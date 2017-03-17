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
#include <numeric>
#include <omp.h>
#include <float.h>

using namespace std;

class Timer
{
public:
    Timer();
    void reset();
    void tic();
    float toc();
    float get();
private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;
};


Timer::Timer()
{
    reset();
}

void Timer::reset()
{
    begin = std::chrono::high_resolution_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(begin-begin);
}

void Timer::tic()
{
    begin = std::chrono::high_resolution_clock::now();
}

float Timer::toc()
{
    duration += std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now()-begin);
    return (float)duration.count()/1000;
}

float Timer::get()
{
    float time = toc();
    tic();
    return time;
}

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
    std::vector<double> W;

    std::vector<double> F;
    std::vector<double> Z;
    std::vector<double> Q;
};
struct Option
{
    Option() : nr_iter(100), nr_lr(0.002), nr_reg(0.002) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter;
    double nr_lr, nr_reg, nr_spl;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: logitboost [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-r <nr_reg>: set the reg\n"
"-s <nr_spl>: set negative sample rate\n");
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
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_spl = stof(args[++i]);
        }
        else
        {
            break;
        }
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command2");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

bool mysortfunc(const pair<string, double>& a, const pair<string, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile(Problem& Tr)
{
        ofstream outfile(opt.Va_out_path);
	for (int f = 0; f < Tr.nr_field; f += 1)
	{
		string fname = Tr.id2featmap[f];
		outfile << fname << " " << Tr.W[f] << endl;
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
		//double s = abs(prob.F[i]);
		double s = prob.F[i];
		userid2score[i] = pair<string, double>(username, s);

		int yi = prob.Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[username] = yi;
	}
	int negcnt = prob.nr_instance - poscnt;
	sort(userid2score.begin(), userid2score.end(), mysortfunc);

	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < userid2score.size(); ++i) 
	{
        	string username = userid2score[i].first;
                double fscore = userid2score[i].second;
		int yi = userid2label[username];
		//cout << i << "\t" << yi << "\t" << fscore << endl;

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
        //istringstream iss(line + "\tconst:1");
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	Tr.Y.push_back(target);
	Tr.X.push_back(std::map<int, double>());

	int i = Tr.X.size() - 1;
	Tr.instidmap[i] = userid;

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		int f = -1;
		if (Tr.feat2idmap.count(feat) == 0) {
			// new feature
			int f = Tr.feat2idmap.size();
			Tr.feat2idmap[feat] = f;
			Tr.id2featmap[f] = feat;
		}
		f = Tr.feat2idmap[feat];
		if (Tr.X[i].count(f) != 0) continue;
		Tr.X[i][f] = x;
        }
    }
    Tr.nr_instance = Tr.X.size();
    Tr.nr_field = Tr.feat2idmap.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{    
    string line;
    while(getline(inputfile, line)) {
        //istringstream iss(line + "\tconst:1");
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	map<int, double> feat2val;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		if (Tr.feat2idmap.count(feat) == 0) continue;
		int j = Tr.feat2idmap[feat];
		feat2val[j] = x;
        }
	
	//有一个非常数项特征在训练数据中出现
	if (feat2val.size() >= 1) {
		Va.Y.push_back(target);

		int i = Va.X.size();
		Va.instidmap[i] = userid;
		
		Va.X.push_back(std::map<int, double>());
		Va.X[i] = feat2val;
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

	Tr.F.resize(Tr.nr_instance, 0);
	Tr.W.resize(Tr.nr_field, 0);

	Tr.Q.resize(Tr.nr_instance, 0);
	Tr.Z.resize(Tr.nr_instance, 0);

	Va.F.resize(Tr.nr_instance, 0);

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

	double PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
	double NegCnt = Tr.Y.size() - PosCnt;
	double F0 =  static_cast<double>(log(PosCnt * 1.0 / NegCnt));

	for (int i = 0; i < Tr.nr_instance; i += 1)
		Tr.F[i] = F0;
	for (int i = 0; i < Va.nr_instance; i += 1)
		Va.F[i] = F0;

	double trainmse = 0;
	double trainllh = 0;
	#pragma omp parallel for schedule(static) reduction(+: trainmse, trainllh)
	for (int i = 0; i < Tr.nr_instance; i += 1) 
	{
		int yi = Tr.Y[i];
		double pi = 1.0 / (1 + exp(-Tr.F[i]));

		Tr.Q[i] = pi * (1 - pi);
		Tr.Z[i] = yi - pi;

		trainmse += (yi - pi) * (yi - pi);
		if (pi != 1 && pi != 0) 
			trainllh += yi * log(pi) + (1 - yi) * log(1 - pi);
	}
	cout << "F0: " << F0 << "\tLLH: " << trainllh / Tr.nr_instance << "\tMSE: " << trainmse / Tr.nr_instance << endl;

	Timer timer;
	map<int, double> featexistmap;
	cout << "iter\ttime\ttr_llh\tva_llh\ttr_auc\tva_auc\ttr_mse\tva_mse\tfeat\tfeatweight" << endl;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		timer.reset();
		timer.tic();
		
		double c = 0;
		vector<double> denominatorvec(Tr.nr_field, 0);
		vector<double> numeratorvec(Tr.nr_field, 0);
		#pragma omp parallel for schedule(dynamic)
		for (int f = 0; f < Tr.nr_field; f += 1)
		{
			if (featexistmap.count(f) > 0) continue;

			for (int j = 0; j < Tr.Xhat[f].size(); j += 1)
			{
				pair<int, double>& ins = Tr.Xhat[f][j];
				int i = ins.first;
				double x = ins.second;
                                numeratorvec[f] += Tr.Z[i] * x;
                                denominatorvec[f] += Tr.Q[i] * x * x;
			}

			feat2llh = 
		}

		double bestfit = 0;
		int besti = 0;
		#pragma omp parallel for schedule(static) 
		for (int i = 0;i < Tr.W.size(); i += 1)
		{
			double fit = (numeratorvec[i] * numeratorvec[i]) / (denominatorvec[i] + opt.nr_reg);
			if (fit > bestfit) 
			{
				bestfit = fit;
				besti = i;
			}
		}
		//double oldweight = Tr.W[besti];
		Tr.W[besti] += opt.nr_lr * numeratorvec[besti] / (denominatorvec[besti] + opt.nr_reg);
		featexistmap[besti] = 1;
		
		double trainmse = 0.0;
		double trainllh = 0.0;
		double trainauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: trainmse, trainllh)
        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
            		int yi = Tr.Y[i];

			double pi = 1.0 / (1 + exp(-Tr.F[i]));
			if (Tr.X[i].count(besti) > 0)
			{
				//Tr.F[i] -= oldweight * Tr.X[i][besti];
				Tr.F[i] += Tr.W[besti] * Tr.X[i][besti];

				pi = 1.0 / (1 + exp(-Tr.F[i]));
				Tr.Q[i] = pi * (1 - pi);
				Tr.Z[i] = yi - pi;
			}

			trainmse += (yi - pi) * (yi - pi);
			if (pi != 1 && pi != 0) 
				trainllh += yi * log(pi) + (1 - yi) * log(1 - pi);
		}
		trainllh /= Tr.nr_instance;
		trainmse /= Tr.nr_instance;
		trainauc = calAUC(Tr);

		double testmse = 0.0;
		double testllh = 0.0;
		double testauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: testmse, testllh)
        	for (int i = 0; i < Va.nr_instance; i += 1) 
		{
            		int yi = Va.Y[i];

			if (Va.X[i].count(besti) != 0)
			{
				//Va.F[i] -= oldweight * Va.X[i][besti];
				Va.F[i] += Tr.W[besti] * Va.X[i][besti];
			}

			double pi = 1.0 / (1 + exp(-Va.F[i]));

			double ctr = 1.0 / (1 + exp(-Va.F[i]));
			ctr = ctr / (ctr + (1 - ctr) / opt.nr_spl);

			testmse += (yi - ctr) * (yi - ctr);
			if (ctr != 1 && ctr != 0) 
				testllh += yi * log(ctr) + (1 - yi) * log(1 - ctr);
		}
		testllh /= Va.nr_instance;
		testmse /= Va.nr_instance;

		testauc = calAUC(Va);
		//writeWeightFile(Tr);

		//cout << iter << "\t" << timer.toc() << "\t" << trainllh << "\t" << testllh << "\t" << trainauc << "\t" << testauc << "\t" << trainmse << "\t" << testmse << endl;
		cout << iter << "\t" << timer.toc() << "\t" << trainllh << "\t" << testllh << "\t" << trainauc << "\t" << testauc << "\t" << trainmse << "\t" << testmse << "\t" << Tr.id2featmap[besti] << "\t" << Tr.W[besti] << endl;
	}

	writeWeightFile(Tr);
return 0;
}
