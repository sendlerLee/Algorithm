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

    std::vector<int> PosIns;
    std::vector<int> NegIns;

    std::vector<double> Y;
    std::vector<double> F;
    std::vector<double> W;
};
struct Option
{
    Option() : nr_iter(100), nr_lr(0.002), nr_sample(2), lambda_2(0.002), lambda_1(0), alpha(0.5) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter, nr_sample;
    double nr_lr;
    double lambda_1, lambda_2;
    double alpha;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: Twitter [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-s <nr_sample>: set the sample count\n"
"-alpha <alpha>: set the llh auc balance\n"
"-l1 <l1_reg>: set the L1 Reg\n"
"-l2 <l2_reg>: set the L2 Reg\n");
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
	else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_sample = stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_lr = stof(args[++i]);
        }
	else if(args[i].compare("-alpha") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.alpha = stof(args[++i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_2 = stof(args[++i]);
        }
	else if(args[i].compare("-l1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_1 = stof(args[++i]);
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
        istringstream iss(line + "\tF0:1");
	string userid;
        int y;
        iss >> userid >> y;
	if (y != 1) y = 0;

	Tr.Y.push_back(y);
	Tr.X.push_back(std::map<int, double>());

	int i = Tr.X.size() - 1;

	if (y == 1) Tr.PosIns.push_back(i);
	else Tr.NegIns.push_back(i);

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		int f = -1;
		if (Tr.feat2idmap.count(feat) == 0) {
			int fid = Tr.feat2idmap.size();
			Tr.feat2idmap[feat] = fid;
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
        istringstream iss(line + "\tF0:1");
	string userid;
        int y;
        iss >> userid >> y;
	if (y != 1) y = 0;

	map<int, double> feat2val;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

		if (Tr.feat2idmap.count(feat) == 0) continue;
		int j = Tr.feat2idmap[feat];
		feat2val[j] = fval;
        }

	//有一个非常数项特征在训练数据中出现
	if (feat2val.size() >= 1) {
		Va.Y.push_back(y);
		Va.F.push_back(0);

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

	Tr.W.resize(Tr.nr_field, 0);
	Tr.F.resize(Tr.nr_instance, 0);

	Timer timer;
	cout << "iter\ttime\ttr_llh\tva_llh\ttr_mse\tva_mse" << endl;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		timer.reset();
		timer.tic();
		srandom(time(NULL));

        	for (int i = 0; i < Tr.nr_instance; i += 1) 
		{
            		int yi = Tr.Y[i];

			float alpha = (rand() % 100) / 100.0;
			//if (alpha > 0.5)
			//if (alpha > 0.2)
			if (alpha > opt.alpha)
			{

				double Fi = 0.0;
				for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
				{
					int f = it->first;
					double x = it->second;
					Fi += Tr.W[f] * x;
				}
				double pi = 1.0 / (1 + exp(- Fi));

				for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it) 
				{
					int f = it->first;
					double x = it->second;
					double g = (yi - pi) * x;
					Tr.W[f] += opt.nr_lr * (g - opt.lambda_2 * Tr.W[f]);
				}
			} else 
			{
				int pi = rand() % Tr.PosIns.size();
				int posidx = Tr.PosIns[pi];

            			map<int, double>& pfdict = Tr.X[posidx];
				double wxi = 0.0;
				for (map<int, double>::iterator it = pfdict.begin(); it != pfdict.end(); ++it) 
				{
					int j = it->first;
					double x = it->second;
					wxi += Tr.W[j] * x;
				}

				for (int s = 0; s < opt.nr_sample; s += 1)
				{
					int ni = rand() % Tr.NegIns.size();
					int negidx = Tr.NegIns[ni];
						
					double wxj = 0.0;
            				map<int, double> ufdict = pfdict;
            				map<int, double>& nfdict = Tr.X[negidx];
					for (map<int, double>::iterator it = nfdict.begin(); it != nfdict.end(); ++it) 
					{
						int j = it->first;
						double x = it->second;
						wxj += Tr.W[j] * x;

						if (pfdict.count(j) > 0) ufdict[j] -= x;
						else ufdict[j] = 0 - x;
					}

					double gradient = - 1.0 / (1 + exp(wxi - wxj));

					for (map<int, double>::iterator it = ufdict.begin(); it != ufdict.end(); ++it) 
					{
						int j = it->first;
						double x = it->second;
						Tr.W[j] -= opt.nr_lr * (gradient * x + opt.lambda_2 * Tr.W[j]);
					}
				}
			}
		}

		int trimcnt = 0;
        	#pragma omp parallel for schedule(static) reduction(+: trimcnt)
		for (int i = 0; i < Tr.W.size(); i += 1)
		{
			if (fabs(Tr.W[i]) < opt.lambda_1) 
			{
				Tr.W[i] = 0;
				trimcnt += 1;
			}
		}

		double trainmse = 0.0;
		double tr_logloss = 0.0;
		double trainauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: trainmse, tr_logloss)
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

			trainmse += (yi - pi) * (yi - pi);
			yi = 2 * yi - 1;
			tr_logloss += log(1 + exp(- yi * Tr.F[i]));
		}
		tr_logloss /= Tr.nr_instance;
		trainmse /= Tr.nr_instance;

		double testmse = 0;
		double va_logloss = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: testmse, va_logloss)
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
			double pi = 1.0 / (1 + exp(-Fi));

			testmse += (yi - pi) * (yi - pi);
			yi = 2 * yi - 1;
			va_logloss += log(1 + exp(- yi * Va.F[i]));
		}
		va_logloss /= Va.nr_instance;
		testmse /= Va.nr_instance;
		double vaauc = calAUC(Va);

		cout << iter << "\t" << timer.toc() << "\t" << tr_logloss << "\t" << va_logloss << "\t" << trainmse << "\t" << testmse << "\t" << vaauc << endl;
	}

	double vaauc = calAUC(Va);
	cout << "validation AUC: " << vaauc << endl;

	writeWeightFile(Tr);
return 0;
}
