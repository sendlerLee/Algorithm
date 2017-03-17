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

#include <limits>
#include <numeric>
#include <thread>
#include <omp.h>

using namespace std;

struct Problem
{
    Problem() {}

    uint32_t nr_instance, nr_field;

    std::map<std::string, uint32_t> feat2idmap;
    std::map<uint32_t, std::string> id2featmap;
    std::map<uint32_t, double> featValSquSum;

    std::map<uint32_t, std::string> instidmap;
    std::vector<map<uint32_t, double>> X;
    std::vector<map<uint32_t, double>> RX;

    std::map<uint32_t, double> G;

    std::vector<double> Y;
    std::vector<double> F;
};

struct Option
{
    Option() : nr_iter(100), nr_lr(0.0001) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter;
    double nr_lr;
};

Option opt;
map<int, double> g_selectFeatMap;

std::string train_help()
{
    return std::string(
"usage: SSFALR [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n");
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
                throw std::invalid_argument("invalid command 1");
            opt.nr_iter = stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command 2");
            opt.nr_lr = stof(args[++i]);
        }
	else break;
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command 3");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

bool mysortfunc(const pair<string, double>& a, const pair<string, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile()
{
	ofstream outfile("feature_weight_ssfalr");
	for (map<int, double>::iterator it = g_selectFeatMap.begin(); it != g_selectFeatMap.end(); ++it)
	{
		int fid = it->first;
		double fval = it->second;
		outfile << fid << " " << fval << endl;
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
	for (int i = 0; i < pvcnt.size(); i += 1)
		cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	writeWeightFile();
	return auc;
}

void loadTrainInstance(Problem& Tr, ifstream& inputfile) 
{    
    string line;
    int idx = 0;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = -1;

	Tr.Y.push_back(target);
	Tr.X.push_back(std::map<uint32_t, double>());

	int i = Tr.X.size() - 1;
	Tr.instidmap[i] = userid;

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		int j = -1;
		if (Tr.feat2idmap.count(feat) == 0) {
			int fid = Tr.feat2idmap.size();
			Tr.feat2idmap[feat] = fid;
			Tr.id2featmap[fid] = feat;

			Tr.RX.push_back(map<uint32_t, double>());
		}
		j = Tr.feat2idmap[feat];
		if (Tr.X[i].count(j) != 0) continue;

		if (Tr.featValSquSum.count(j) == 0) Tr.featValSquSum[j] = 0;
		Tr.featValSquSum[j] += fval * fval;

		Tr.X[i][j] = fval;
		Tr.RX[j][i] = fval;
        }
    }
    Tr.nr_instance = Tr.X.size();
    Tr.nr_field = Tr.RX.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{    
    string line;
    int idx = 0;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = -1;

	Va.Y.push_back(target);
	Va.X.push_back(std::map<uint32_t, double>());

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

int main(int const argc, char const * const * const argv) 
{
	try {
		opt = parse_option(argv_to_args(argc, argv));
	}
	catch(std::invalid_argument const &e) {	
		std::cout << e.what();
		return EXIT_FAILURE;
	}

	Problem Tr, Va;

	ifstream trainfile(opt.Tr_path);
	loadTrainInstance(Tr, trainfile);
	cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field <<  endl;

	ifstream testfile(opt.Va_path);
	loadTestInstance(Tr, Va, testfile);
	cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;

	double x = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
	double y = Tr.Y.size();

	int N_Pos = (y + x) / 2;
	int N_Neg = (y - x) / 2;
	double bias = static_cast<double>(log(N_Pos * 1.0 / N_Neg));
	cout << "bias: " << bias << "\tN_Pos: " << N_Pos << "\tN_Neg: " << N_Neg << endl;

	double trainllk = 0;
	for (int i = 0; i < Tr.nr_instance; i += 1)
	{
		Tr.F.push_back(bias);
		Tr.G[i] = (Tr.Y[i] - 1.0 / (1 + exp(- Tr.F[0])));
		trainllk += log(1 + exp(- Tr.Y[i] * Tr.F[i]));
	}
	trainllk /= Tr.nr_instance;

	double testllk = 0;
	for (int i = 0; i < Va.nr_instance; i += 1)
	{
		Va.F.push_back(bias);
		testllk += log(1 + exp(- Va.Y[i] * Va.F[i]));
	}
	testllk /= Va.nr_instance;
	cout << "init Train_LLK: " << trainllk << "\tTest_LLK: " << testllk << endl;

  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		double minError = 1000000;
		int bestj = -1;
		double bestwgt = 0;
		for (int j = 0; j < Tr.nr_field; j += 1)
		{
			if (g_selectFeatMap.count(j) != 0)continue;

			//first-order derivative
			double fod = 0;
			//Second-order derivative
			double sod = 0;

			double b = 0;
			for (map<uint32_t, double>::iterator it = Tr.RX[j].begin(); it != Tr.RX[j].end(); ++it)
			{
				int i = it->first;
				double x = it->second;
				b += Tr.G[i] * x;
			
				int y = Tr.Y[i];
				double pi = 1.0 / (1 + exp(y * Tr.F[i]));
				fod += - y * x * pi;
				sod += x * x * pi * (1 - pi);
			}
			double a = Tr.featValSquSum[j];

			double wi = b / a;
			double error = - wi * b;

			wi = - fod / sod;
			//error = - fod * fod / (2 * sod);

			if (error < minError)
			{
				minError = error;
				bestj = j;
				bestwgt = wi;
			}
			cout << "Iter: "<< iter << "\tError: " << error << "\tFeat: " << Tr.id2featmap[j] << "\tWeight: " << wi << "\tfod: " << fod << "\tsod: " << sod  << endl;
		}

		g_selectFeatMap[bestj] = bestwgt;
		cout << "Iter: "<< iter << "\tBestFeat: " << bestj << "\tFeat: " << Tr.id2featmap[bestj] << "\tWeight: " << g_selectFeatMap[bestj] << "\tError: " << minError << endl;

		double trainllk = 0;
		//#pragma omp parallel for schedule(static) reduction(+: trainllk)
		for (map<uint32_t, double>::iterator it = Tr.RX[bestj].begin(); it != Tr.RX[bestj].end(); ++it)
		{
			int i = it->first;
			double yi = Tr.Y[i];
			double x = it->second;

			Tr.F[i] += opt.nr_lr * bestwgt * x;
			Tr.G[i] = (yi - 1.0 / (1 + exp(- Tr.F[i])));

			trainllk += log(1 + exp(- yi * Tr.F[i]));
		}
		trainllk /= Tr.nr_instance;

		//double auc2 = calAUC(Tr);
		//cout << "Iter: " << iter << "\tTrain AUC: " << auc2 << endl << endl;

		double testllk = 0;
		#pragma omp parallel for schedule(static) reduction(+: testllk)
		for (int i = 0; i < Va.nr_instance; i += 1)
		{
			if (Va.X[i].count(bestj) != 0)
				Va.F[i] += opt.nr_lr * bestwgt * Va.X[i][bestj];
			testllk += log(1 + exp(- Va.Y[i] * Va.F[i]));
		}
		testllk /= Va.nr_instance;
		cout << "Train_LLK: " << trainllk << "\tTest_LLK: " << testllk << endl;

		double auc = calAUC(Va);
		cout << "Iter: " << iter << "\tTest AUC: " << auc << endl << endl;
	}

return 0;
}
