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
#include <iomanip>

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
    std::vector<double> Q;

    double F0;
};

struct Option
{
	Option() : nr_iter(100), nr_lr(0.01), lambda_2(100), print_auc(0), lambda_1(0.0), algorithm(3), br(1.0) {}
	std::string Tr_path, Va_path, Va_out_path;
	int nr_iter;
	int print_auc;
	int algorithm;
	double nr_lr, lambda_2;
	double lambda_1;
	double br;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: logitboost [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate default 0.01\n"
"===========High Order Usage===========\n"
"-alg <algorithm>: 1: Gaussian; 2: AdaBoost; 3: Bernoulli; 4: Poisson; 5 : balance Bernoulli default 3 \n"
"-a <print_auc>: print test auc default 0 \n"
"-b <blance rate>: set balance rate only valid for alg 5 default 1 \n"
"-l2 <lambda_2>: set the reg default 100 \n"
"-l1 <L1_Reg>: L1 Reg default 0 \n");
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
        else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_2 = stof(args[++i]);
        }
	else if(args[i].compare("-a") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.print_auc = stoi(args[++i]);
        }
	else if(args[i].compare("-alg") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.algorithm = stoi(args[++i]);
        }
	else if(args[i].compare("-l1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_1 = stof(args[++i]);
        }
	else if(args[i].compare("-b") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.br = stof(args[++i]);
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

void writeWeightFile(Problem& Tr)
{
        ofstream outfile(opt.Va_out_path);
	outfile << "F0 " << Tr.F0 << endl;
	for (map<string, int>::iterator it = Tr.feat2idmap.begin(); it != Tr.feat2idmap.end(); ++it)
	{
		string feat = it->first;
		int j = it->second;
		outfile << feat << " " << Tr.W[j] << endl;
        }
	outfile.close();
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

		double yi = prob.Y[i];
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
	while(getline(inputfile, line)) 
	{
		istringstream iss(line);
		string userid;
		int target;
		iss >> userid >> target;
		if (target != 1) target = 0;
		
		Tr.Y.push_back(target);
		Tr.X.push_back(std::map<int, double>());

		int i = Tr.X.size() - 1;
		string feature;
		while (iss) 
		{
			iss >> feature;
			int findex = feature.find_last_of(":");
			string feat = feature.substr(0, findex).c_str();
			double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			int f = -1;
			if (Tr.feat2idmap.find(feat) == Tr.feat2idmap.end())
			{
				int f = Tr.feat2idmap.size();
				Tr.feat2idmap[feat] = f;
			}
			f = Tr.feat2idmap[feat];
			Tr.X[i][f] = x;
		}
	}
	Tr.nr_instance = Tr.X.size();
	Tr.nr_field = Tr.feat2idmap.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{    
	string line;
	while(getline(inputfile, line)) 
	{
		istringstream iss(line);
		string userid;
		int target;
		iss >> userid >> target;
		if (target != 1) target = 0;
		
		map<int, double> feat2val;
		string feature;
		while (iss) 
		{
			iss >> feature;
			int findex = feature.find_last_of(":");
			string feat = feature.substr(0, findex).c_str();
			if (Tr.feat2idmap.count(feat) == 0) continue;

			double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			int j = Tr.feat2idmap[feat];
			feat2val[j] = x;
		}
	
		if (feat2val.size() >= 1) 
		{
			Va.Y.push_back(target);
			int i = Va.X.size();
		
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

	Tr.Z.resize(Tr.nr_instance, 0);
	Tr.Q.resize(Tr.nr_instance, 0);

	Va.F.resize(Va.nr_instance, 0);

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

	//Gaussian
	if (opt.algorithm == 1)
		Tr.F0 = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0) * 1.0 / Tr.Y.size();

	//AdaBoost
	if (opt.algorithm == 2)
	{
		double PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
		double NegCnt = Tr.Y.size() - PosCnt;
		Tr.F0 = static_cast<double>(log(PosCnt * 1.0 / NegCnt) / 2.0);
	}

	//Bernoulli
	if (opt.algorithm == 3)
	{
		double PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
		double NegCnt = Tr.Y.size() - PosCnt;
		Tr.F0 =  static_cast<double>(log(PosCnt * 1.0 / NegCnt));
	}
				
	//Poisson
	if (opt.algorithm == 4)
	{
		double PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
		Tr.F0 =  static_cast<double>(log(PosCnt * 1.0 / Tr.Y.size()));
	}

	//unbalance lr
	//L = y * (F - (t - 1) log(1 + exp(F))) - t * log(1 + exp(F));
	double br = 1;
	if (opt.algorithm == 5)
	{
		double PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0);
		double NegCnt = Tr.Y.size() - PosCnt;
		if (opt.br == 1) br = PosCnt * 1.0 / NegCnt;
		else br = opt.br;
		Tr.F0 =  static_cast<double>(log(PosCnt * 1.0 / (br * NegCnt)));
	}

	for (int i = 0; i < Tr.nr_instance; i += 1)
		Tr.F[i] = Tr.F0;
	for (int i = 0; i < Va.nr_instance; i += 1)
		Va.F[i] = Tr.F0;

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Tr.nr_instance; i += 1) 
	{
		double yi = Tr.Y[i];

		if (opt.algorithm == 1)
		{
			Tr.Z[i] = yi - Tr.F[i];
			Tr.Q[i] = 1;
		}

		if (opt.algorithm == 2)
		{
			Tr.Z[i] = (2 * yi - 1) * exp(- (2 * yi - 1) * Tr.F[i]);
			Tr.Q[i] = exp(- (2 * yi - 1) * Tr.F[i]);
		}

		if (opt.algorithm == 3)
		{
			double pi = 1.0 / (1 + exp(- Tr.F[i]));
			Tr.Z[i] = yi - pi;
			Tr.Q[i] = pi * (1 - pi);
		}

		if (opt.algorithm == 4)
		{
			Tr.Z[i] = yi - exp(Tr.F[i]);
			Tr.Q[i] = exp(Tr.F[i]);
		}

		if (opt.algorithm == 5)
		{
			double pi = 1.0 / (1 + exp(- Tr.F[i]));
			Tr.Z[i] = yi + yi * (br - 1) * pi - br * pi;
			Tr.Q[i] = - (yi * (br - 1) - br) * pi * (1 - pi);
		}
	}

	double va_loss = 0.0;
	double va_mse = 0.0;
	double va_logloss = 0.0;
	#pragma omp parallel for schedule(static) reduction(+: va_loss, va_mse, va_logloss)
        for (int i = 0; i < Va.nr_instance; i += 1) 
	{
            	double yi = Va.Y[i];
		double pi = -1;
	
		if (opt.algorithm == 1)
		{
			va_loss += (yi - Va.F[i]) * (yi - Va.F[i]);
			pi = Va.F[i];
		}

		if (opt.algorithm == 2)
		{
			va_loss += exp(-(2 * yi - 1) * Va.F[i]);
			pi = 1.0 / (1 + exp(- 2 * Va.F[i]));
		}

		if (opt.algorithm == 3)
		{
			va_loss -= (yi * Va.F[i] - log(1 + exp(Va.F[i])));
			pi = 1.0 / (1 + exp(- Va.F[i]));
		}

		if (opt.algorithm == 4)
		{
			va_loss -= (yi * Va.F[i] - exp(Va.F[i]));
			pi = exp(Va.F[i]);
		}
	
		if (opt.algorithm == 5)
		{
			va_loss -= (yi * Va.F[i] + (yi * (br - 1) - br) * log(1 + exp(Va.F[i])));
			pi = 1.0 / (1 + exp(- Va.F[i]));
		}

		if (pi > 0.999) pi = 0.999;
		if (pi < 0.001) pi = 0.001;
		va_mse += (yi - pi) * (yi - pi);
		va_logloss -= yi * log(pi) + (1 - yi) * log(1 - pi);
	}
	va_loss /= Va.nr_instance;
	va_mse /= Va.nr_instance;
	va_logloss /= Va.nr_instance;

	cout << "iter\ttime\tloss\tlogloss\tmse";
	if (opt.print_auc == 1) cout << "\tauc";
	if (opt.lambda_1 > 0) cout << "\tsparsity";
	cout << endl;

	cout << "Init F0:\t" << Tr.F0 << endl;
	cout << "-1\t0\t" << va_loss << "\t" << va_mse << endl;

	Timer timer;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		timer.reset();
		timer.tic();

		vector<double> denominatorvec(Tr.nr_field, 0);
		vector<double> numeratorvec(Tr.nr_field, 0);
		#pragma omp parallel for schedule(dynamic)
		for (int f = 0; f < Tr.nr_field; f += 1)
		{
			for (int j = 0; j < Tr.Xhat[f].size(); j += 1)
			{
				pair<int, double>& ins = Tr.Xhat[f][j];
				int i = ins.first;
				double x = ins.second;
                                numeratorvec[f] += Tr.Z[i] * x;
                                denominatorvec[f] += Tr.Q[i] * x * x;
			}
		}

                int trimcnt = 0;
                #pragma omp parallel for schedule(static) reduction(+: trimcnt)
                for (int i = 0;i < Tr.W.size(); i += 1)
                {
                        Tr.W[i] += opt.nr_lr * numeratorvec[i] / (denominatorvec[i] + opt.lambda_2);
                        if (opt.lambda_1 > 0 && fabs(Tr.W[i]) < opt.lambda_1)
                        {
                                Tr.W[i] = 0;
                                trimcnt += 1;
                        }
                }
		double sparity = trimcnt * 100.0 / Tr.nr_field;

		#pragma omp parallel for schedule(static)
        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
            		double yi = Tr.Y[i];
			double Fi = 0.0;

			for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Tr.F[i] = Fi + Tr.F0;

			if (opt.algorithm == 1)
			{
				Tr.Z[i] = yi - Tr.F[i];
				Tr.Q[i] = 1;
			}

			if (opt.algorithm == 2)
			{
				Tr.Z[i] = (2 * yi - 1) * exp(- (2 * yi - 1) * Tr.F[i]);
				Tr.Q[i] = exp(- (2 * yi - 1) * Tr.F[i]);
			}

			if (opt.algorithm == 3)
			{
				double pi = 1.0 / (1 + exp(- Tr.F[i]));
				Tr.Z[i] = yi - pi;
				Tr.Q[i] = pi * (1 - pi);
			}

			if (opt.algorithm == 4)
			{
				Tr.Z[i] = yi - exp(Tr.F[i]);
				Tr.Q[i] = exp(Tr.F[i]);
			}

			if (opt.algorithm == 5)
			{
				double pi = 1.0 / (1 + exp(- Tr.F[i]));
				Tr.Z[i] = yi + yi * (br - 1) * pi - br * pi;
				Tr.Q[i] = - (yi * (br - 1) - br) * pi * (1 - pi);
			}
		}

		double va_loss = 0.0;
		double va_mse = 0.0;
		double va_logloss = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: va_loss, va_mse, va_logloss)
        	for (int i = 0; i < Va.nr_instance; i += 1) 
		{
            		double yi = Va.Y[i];

			double Fi = 0.0;
			for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it) 
			{
				int f = it->first;
				double x = it->second;
				Fi += Tr.W[f] * x;
			}
			Va.F[i] = Fi + Tr.F0;
			double pi = -1;

			if (opt.algorithm == 1)
			{
				va_loss += (yi - Va.F[i]) * (yi - Va.F[i]);
				pi = Va.F[i];
			}

			if (opt.algorithm == 2)
			{
				va_loss += exp(-(2 * yi - 1) * Va.F[i]);
				pi = 1.0 / (1 + exp(- 2 * Va.F[i]));
			}

			if (opt.algorithm == 3)
			{
				va_loss -= (yi * Va.F[i] - log(1 + exp(Va.F[i])));
				pi = 1.0 / (1 + exp(- Va.F[i]));
			}
	
			if (opt.algorithm == 4)
			{
				va_loss -= (yi * Va.F[i] - exp(Va.F[i]));
				pi = exp(Va.F[i]);
			}

			if (opt.algorithm == 5)
			{
				va_loss -= (yi * Va.F[i] + (yi * (br - 1) - br) * log(1 + exp(Va.F[i])));
				pi = 1.0 / (1 + exp(- Va.F[i]));
			}

			if (pi > 0.999) pi = 0.999;
			if (pi < 0.001) pi = 0.001;
			va_mse += (yi - pi) * (yi - pi);
			va_logloss -= yi * log(pi) + (1 - yi) * log(1 - pi);
		}
		va_loss /= Va.nr_instance;
		va_logloss /= Va.nr_instance;
		va_mse /= Va.nr_instance;

		cout << iter << "\t" << timer.toc() << "\t" << va_loss << "\t" << va_logloss << "\t" << va_mse;
		if (opt.print_auc == 1)
		{
			double vaauc = calAUC(Va);
			cout << "\t" << vaauc;
		}

		if (opt.lambda_1 > 0)
		{
			cout << "\t" << sparity << "%";
		}

		double sumposctr = 0, sumnegctr = 0;
		int sumposcnt = 0, sumnegcnt = 0;
		#pragma omp parallel for schedule(static) reduction(+: sumposctr, sumnegctr, sumposcnt, sumnegcnt)
        	for (int i = 0; i < Va.nr_instance; i += 1) 
		{
            		double yi = Va.Y[i];
			double pi = -1;
			if (opt.algorithm == 1)
				pi = Va.F[i];

			if (opt.algorithm == 2)
				pi = 1.0 / (1 + exp(- 2 * Va.F[i]));

			if (opt.algorithm == 3)
				pi = 1.0 / (1 + exp(- Va.F[i]));
	
			if (opt.algorithm == 4)
				pi = exp(Va.F[i]);

			if (opt.algorithm == 5)
				pi = 1.0 / (1 + exp(- Va.F[i]));
	
			if (yi > 0) 
			{
				sumposctr += pi;
				sumposcnt += 1;
			} else 
			{
				sumnegctr += pi;
				sumnegcnt += 1;
			}
		}

		cout << "\t" << sumposctr * 1.0 / sumposcnt << "\t" << sumnegctr * 1.0 / sumnegcnt;

		cout << endl;
	}


	double vaauc = calAUC(Va);
	cout << "validation AUC: " << vaauc << endl;
	writeWeightFile(Tr);
return 0;
}
