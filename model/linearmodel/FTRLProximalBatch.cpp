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

    std::vector<double> Y;
    std::vector<double> W;

    std::vector<double> F;

    std::vector<double> Z;
    std::vector<double> N;
};

struct Option
{
    Option() : nr_iter(1){}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter;
    double alpha, beta, lambda_1, lambda_2;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: FTRLProximalBatch [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration default value 1\n"
"-a <alpha>: set alpha\n"
"-b <beta>: set beta\n"
"-l1 <lambda_1>: lambda_1\n"
"-l2 <lambda_2>: lambda_2\n");
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
        else if(args[i].compare("-a") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.alpha = stod(args[++i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.beta = stod(args[++i]);
        }
	else if(args[i].compare("-l1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_1 = stod(args[++i]);
        }
	else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_2 = stod(args[++i]);
        }
        else
            break;
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
	istringstream iss(line + "\tF0:1");
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
	istringstream iss(line + "\tF0:1");
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

int sign(double x)
{
	if(x < 0) 
		return -1;
	else
		return 1;
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
	
	omp_set_num_threads(50);

	Problem Tr, Va;

	ifstream trainfile(opt.Tr_path);
	loadTrainInstance(Tr, trainfile);
	cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field <<  endl;

	ifstream testfile(opt.Va_path);
	loadTestInstance(Tr, Va, testfile);
	cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;

	Tr.F.resize(Tr.nr_instance, 0.0);
	Va.F.resize(Va.nr_instance, 0.0);

	Tr.W.resize(Tr.nr_field, 0.0);
	Tr.N.resize(Tr.nr_field, 0.0);
	Tr.Z.resize(Tr.nr_field, 0.0);

	double alpha = opt.alpha;
	double beta = opt.beta;
	double lambda_1 = opt.lambda_1;
	double lambda_2 = opt.lambda_2;

	Timer timer;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		timer.reset();
		timer.tic();

		#pragma omp parallel for schedule(static) 
        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
			double Fi = 0;
			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
			{
				int j = it->first;
				if (abs(Tr.Z[j]) <= lambda_1)
					Tr.W[j] = 0;
				else 
					Tr.W[j] = - (Tr.Z[j] - sign(Tr.Z[j]) * lambda_1) / ((beta + sqrt(Tr.N[j])) / alpha + lambda_2);

				double x = it->second;
				Fi += Tr.W[j] * x;
			}
			double pi = 1.0 / (1 + exp(- Fi));
			
			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
			{
				int j = it->first;
				double x = it->second;
				double g = (pi - Tr.Y[i]) * x;
				double tao = (sqrt(Tr.N[j] + g * g) - sqrt(Tr.N[j])) / alpha;
				Tr.Z[j] += g - tao * Tr.W[j];
				Tr.N[j] += g * g;
			}
		 }

                int trimcnt = 0;
                #pragma omp parallel for schedule(static) reduction(+: trimcnt)
                for (int i = 0;i < Tr.W.size(); i += 1)
                {
                        if (abs(Tr.W[i]) <= 0)
                                trimcnt += 1;
                }
		double sparity = trimcnt * 100.0 / Tr.nr_field;

		double tr_mse = 0.0;
		double tr_logloss = 0.0;
		double trainauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: tr_mse, tr_logloss)
        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
            		int yi = Tr.Y[i];
			double Fi = 0.0;
			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Tr.F[i] = Fi;
			double pi = 1.0 / (1 + exp(-Tr.F[i]));

			tr_mse += (yi - pi) * (yi - pi);
			yi = 2 * yi - 1;
			tr_logloss += log(1 + exp(- yi * Tr.F[i]));
		}
		tr_logloss /= Tr.nr_instance;
		tr_mse /= Tr.nr_instance;

		double va_mse = 0.0;
		double va_logloss = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: va_mse, va_logloss)
        	for (int i = 0; i < Va.nr_instance; i += 1) 
		{
            		int yi = Va.Y[i];

			double Fi = 0.0;
			for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Va.F[i] = Fi;
			double pi = 1.0 / (1 + exp(-Va.F[i]));

			va_mse += (yi - pi) * (yi - pi);
			yi = 2 * yi - 1;
			va_logloss += log(1 + exp(- yi * Va.F[i]));
		}
		va_logloss /= Va.nr_instance;
		va_mse /= Va.nr_instance;

		cout << iter << "\t" << timer.toc() << "\t" << tr_logloss << "\t" << va_logloss << "\t" << tr_mse << "\t" << va_mse << "\t" << sparity << endl;
	}

	double vaauc = calAUC(Va);
	cout << "validation AUC: " << vaauc << endl;
	writeWeightFile(Tr);
return 0;
}
