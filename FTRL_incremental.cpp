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

//author: tannaiqiang (532429163@qq.com)

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

    std::vector<map<int, double>> X;
    std::vector<vector<pair<int, double>>> Xhat;

    std::vector<double> Y;
    std::vector<double> W;

    std::vector<double> F;
    std::vector<double> Z;
    std::vector<double> N;

    double F0;
};
struct Option
{
    Option() : alpha(100), beta(1), r1(1), r2(1) {}
    std::string Tr_path, Va_path, Va_out_path, state_path, previous_state_path = "none";
    int print_auc = 1;
    int iterative = 0;
    double alpha, beta, r1, r2;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: FTRL_incremental [<options>] <train_path> <validation_path> <validation_output_path> <state_path>\n"
"\n"
"options:\n"
"-a <param_alpha>: default 0.1\n"
"-b <param_beta>: default 1.0\n"
"-r1 <param_r1>: default 100 \n"
"-r2 <param_r2>: default 100 \n"
"-r2 <param_iterative>: default 0 (false)\n"
"-p <previous_state_path> \n"
);
}

Option parse_option(std::vector<std::string> const &args)
{
    int const argc = static_cast<int>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    int i = 0;
    for(; i < argc; ++i)
    {
	if(args[i].compare("-a") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.alpha = stof(args[++i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.beta = stof(args[++i]);
        }
        else if(args[i].compare("-r1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.r1 = stof(args[++i]);
        }
	else if(args[i].compare("-r2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.r2 = stof(args[++i]);
        }
	else if(args[i].compare("-i") == 0)
	{
	    if (i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.iterative = stoi(args[++i]);
	}
	else if(args[i].compare("-p") == 0)
	{
	    if (i == argc-1)
		throw std::invalid_argument("invalid command");
	    opt.previous_state_path = args[++i];
	}
        else
            break;
    }

    if(i != argc-4) {
	cout<<argc-i<<endl;
        throw std::invalid_argument("invalid command2");
    }

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];
    opt.state_path = args[i++];

    return opt;
}

void writeWeightFile(Problem& Tr)
{
        ofstream outfile(opt.Va_out_path);
	ofstream statefile(opt.state_path);
	for (map<string, int>::iterator it = Tr.feat2idmap.begin(); it != Tr.feat2idmap.end(); ++it)
	{
		string feat = it->first;
		int j = it->second;
		if (Tr.W[j] != 0) outfile << feat << " " << Tr.W[j] << endl;
		statefile << "state_Z_" + feat << " " << Tr.Z[j] << endl;
		statefile << "state_N_" + feat << " " << Tr.N[j] << endl;
        }
	outfile.close();
	statefile.close();
}

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

double calAUC(Problem& prob)
{
	int poscnt = 0;
	vector<pair<int, double>> userid2score(prob.nr_instance);
	map<int, double> userid2label;
	for (int i = 0; i < prob.nr_instance; i += 1)
	{
		double s = prob.F[i];
		userid2score[i] = pair<int, double>(i, s);

		int yi = prob.Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[i] = yi;
	}
	int negcnt = prob.nr_instance - poscnt;
	sort(userid2score.begin(), userid2score.end(), mysortfunc);

	double score = 0;
	int predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < userid2score.size(); ++i) 
	{
        	int idx = userid2score[i].first;
                double fscore = userid2score[i].second;
		int yi = userid2label[idx];

		if (yi == 1) 
		{
			score += belownegcnt * 1.0 / negcnt;
			predictposcnt += 1;
		} 
		else {
			belownegcnt -= 1;
		}
	}
	
	double auc = score * 1.0 / poscnt;
	return auc;
}

void loadTrainInstance(Problem& Tr, ifstream& inputfile) 
{
    string line;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	Tr.Y.push_back(target);
	Tr.X.push_back(std::map<int, double>());

	int i = Tr.X.size() - 1;
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
		}
		f = Tr.feat2idmap[feat];
		if (Tr.X[i].count(f) != 0) continue;
		Tr.X[i][f] = x;
        }
	string feat = "F0";
	if (Tr.feat2idmap.count(feat) == 0) {
        	int f = Tr.feat2idmap.size();
		Tr.feat2idmap[feat] = f;
	}
	int f = Tr.feat2idmap[feat];
	Tr.X[i][f] = 1;
    }
    Tr.nr_instance = Tr.X.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{    
    string line;
    while(getline(inputfile, line)) {
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

		if (Tr.feat2idmap.count(feat) == 0) {
			//continue;
			int f = Tr.feat2idmap.size();
			Tr.feat2idmap[feat] = f;
		}
		int j = Tr.feat2idmap[feat];
		feat2val[j] = x;
        }
	string feat = "F0";
	if (Tr.feat2idmap.count(feat) != 0)
	{
		int j = Tr.feat2idmap[feat];
		feat2val[j] = 1;
	}
	if (feat2val.size() >= 1) {
		Va.Y.push_back(target);
		int i = Va.X.size();
		
		Va.X.push_back(std::map<int, double>());
		Va.X[i] = feat2val;
	}
    }
    Va.nr_instance = Va.X.size();
}

void loadPreviousFeature(Problem& Tr, Problem& Va, ifstream& inputfile)
{
    string line;
    while(getline(inputfile, line)) {
        istringstream iss(line);
        std::string feat;
        double weight;
        iss >> feat >> weight;

        string prefix = feat.substr(0, 8);
        if (prefix == "state_Z_") {
            feat = feat.substr(8);
            if (Tr.feat2idmap.count(feat) == 0) {
                int f = Tr.feat2idmap.size();
                Tr.feat2idmap[feat] = f;
	    }
        }
	else if (prefix == "state_N_") {
            feat = feat.substr(8);
            if (Tr.feat2idmap.count(feat) == 0) {
                int f = Tr.feat2idmap.size();
                Tr.feat2idmap[feat] = f;
	    }
        }
    }
}

void loadPreviousState(Problem& Tr, Problem& Va, ifstream& inputfile)
{
    string line;
    while(getline(inputfile, line)) {
        istringstream iss(line);
        std::string feat;
        double weight;
        iss >> feat >> weight;
	
	string prefix = feat.substr(0, 8);
	if (prefix == "state_Z_") {
	    feat = feat.substr(8);
	    if (Tr.feat2idmap.count(feat) > 0) {
                int f = Tr.feat2idmap[feat];
                Tr.Z[f] = weight;
    	    }
  	}else if (prefix == "state_N_") {
	    feat = feat.substr(8);
            if (Tr.feat2idmap.count(feat) > 0) {
                int f = Tr.feat2idmap[feat];
                Tr.N[f] = weight;
            }
    	}
    }
    for (int f = 0; f < Tr.nr_field; f += 1)
    {
	if (fabs(Tr.Z[f]) <= opt.r1)
        {
        	Tr.W[f] = 0;
  	}
	else
	{
		double A = (opt.beta + sqrt(Tr.N[f])) / opt.alpha + opt.r2;
        	double B = Tr.Z[f] - opt.r1;
                if (Tr.Z[f] < 0) {
                	B = Tr.Z[f] + opt.r1;
       		}
		else if (Tr.Z[f] == 0) {
			B = 0;
		}
                Tr.W[f] = -B/A;
       	}
    }
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

	if (opt.previous_state_path != "none"){
        	ifstream statefile(opt.previous_state_path);
        	loadPreviousFeature(Tr, Va, statefile);
        }
	Tr.nr_field = Tr.feat2idmap.size();

	Tr.F.resize(Tr.nr_instance, 0);
	Va.F.resize(Va.nr_instance, 0);

	Tr.W.resize(Tr.nr_field, 0);
	Tr.Z.resize(Tr.nr_field, 0);
	Tr.N.resize(Tr.nr_field, 0);

	if (opt.previous_state_path != "none"){
        	ifstream statefile(opt.previous_state_path);
        	loadPreviousState(Tr, Va, statefile);
        }

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

	for (int i = 0; i < Tr.nr_instance; i += 1) {
		double Fi = 0;
		for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
                {
                        int f = it->first;
                        double x = it->second;
                        if (fabs(Tr.Z[f]) <= opt.r1) 
			{
			    Tr.W[f] = 0;
			}
			else 
			{
			    double A = (opt.beta + sqrt(Tr.N[f])) / opt.alpha + opt.r2;
			    double B = Tr.Z[f] - opt.r1; 
			    if (Tr.Z[f] < 0) {
				B = Tr.Z[f] + opt.r1;
			    }
			    else if (Tr.Z[f] == 0) {
				B = 0;
			    }
			    Tr.W[f] = -B/A;
			}
			Fi += Tr.W[f] * x; 
                }
		Tr.F[i] = Fi;
		//update state
		for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
		{
                        int f = it->first;
                        double x = it->second;
			double predict = 1.0 / (1 + exp(-Tr.F[i]));
			double gradient = (predict - Tr.Y[i]) * x;  
			double sigma = (sqrt(Tr.N[f] + gradient * gradient) - sqrt(Tr.N[f])) / opt.alpha;
			Tr.Z[f] = Tr.Z[f] + gradient - sigma * Tr.W[f];
			Tr.N[f] = Tr.N[f] + gradient * gradient;
                }
	}

	for (int i = 0; i < Va.nr_instance; i += 1) {
        	double Fi = 0.0;
                for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it)
                {
                	int f = it->first;
                        double x = it->second;
			if (opt.iterative)
			{
				if (fabs(Tr.Z[f]) <= opt.r1)
                        	{
                            		Tr.W[f] = 0;
                        	}
                        	else
                        	{
                            		double A = (opt.beta + sqrt(Tr.N[f])) / opt.alpha + opt.r2;
                            		double B = Tr.Z[f] - opt.r1;
                            		if (Tr.Z[f] < 0) {
                                		B = Tr.Z[f] + opt.r1;
                            		} 
					else if (Tr.Z[f] == 0) {
						B = 0;
					}
                            		Tr.W[f] = -B/A;
                        	}
			}
                	Fi += Tr.W[f] * x;
		}
               	Va.F[i] = Fi;
		if (opt.iterative)
		{
			//update state
			for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it)
                	{
                        	int f = it->first;
                        	double x = it->second;
                        	double predict = 1.0 / (1 + exp(-Va.F[i]));
                        	double gradient = (predict - Va.Y[i]) * x;
                        	double sigma = (sqrt(Tr.N[f] + gradient * gradient) - sqrt(Tr.N[f])) / opt.alpha;
                        	Tr.Z[f] = Tr.Z[f] + gradient - sigma * Tr.W[f];
                        	Tr.N[f] = Tr.N[f] + gradient * gradient;
			}
                }
	}
	double tr_mse = 0;
	double tr_logloss = 0;
	#pragma omp parallel for schedule(static) reduction(+: tr_mse, tr_logloss)
	for (int i = 0; i < Tr.nr_instance; i += 1) 
	{
		int yi = Tr.Y[i];
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
		double pi = 1.0 / (1 + exp(-Va.F[i]));

		va_mse += (yi - pi) * (yi - pi);
		yi = 2 * yi - 1;
		va_logloss += log(1 + exp(- yi * Va.F[i]));
	}
	va_logloss /= Va.nr_instance;
	va_mse /= Va.nr_instance;

	if (opt.print_auc == 1)
		cout << "iter\ttime\ttr_logloss\tva_logloss\ttr_mse\tva_mse\tva_auc\tsparity" << endl;
	else if (opt.print_auc == 0)
		cout << "iter\ttime\ttr_logloss\tva_logloss\ttr_mse\tva_mse\tsparity" << endl;

	cout << "-1\t0\t" << tr_logloss << "\t" << va_logloss << "\t" << tr_mse << "\t" << va_mse << endl;

	double vaauc = calAUC(Va);
	cout << "validation AUC: " << vaauc << endl;
	writeWeightFile(Tr);
	return 0;
}
