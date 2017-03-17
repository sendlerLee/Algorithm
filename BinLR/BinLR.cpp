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
    std::vector<double> F;
    std::vector<double> W;
};
struct Option
{
    Option() : nr_iter(100), nr_lr(0.002), nr_reg(0.002) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter;
    double nr_lr, nr_reg;
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

bool mysortfunc(const pair<string, double>& a, const pair<string, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile(Problem& Tr)
{
	ofstream outfile("feature_weight_binlr");
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
                outfile << username << "\t" << yi << "\t" << fscore << endl;

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
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	Tr.Y.push_back(target);
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
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	Va.Y.push_back(target);
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

	vector<int> trainidxvec(Tr.nr_instance);
        for (int idx = 0; idx < Tr.nr_instance; idx += 1) 
		trainidxvec[idx] = idx;

	Timer timer;
	cout << "iter\ttime\ttr_llh\tva_llh\ttr_auc\tva_auc\ttr_mse\tva_mse" << endl;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		timer.tic();
		std::random_shuffle(trainidxvec.begin(), trainidxvec.end());
        	for (int idx = 0; idx < trainidxvec.size(); idx += 1) 
        	//for (int i = 0; i < Tr.X.size(); i += 1)
		{
			int i = trainidxvec[idx];
            		int yi = Tr.Y[i];

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
				double g = yi * x - x * pi;
				Tr.W[f] += opt.nr_lr * (g - opt.nr_reg * Tr.W[f]);
			}
		}

		double trainmse = 0.0;
		double trainllk = 0.0;
		double trainauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: trainmse, trainllk)
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
			Tr.F[i] = 1.0 / (1 + exp(-Fi));
			trainllk += yi * Fi - log(1 + exp(Fi));

			trainmse += (yi - Tr.F[i]) * (yi - Tr.F[i]);
		}
		trainllk /= Tr.nr_instance;
		trainmse /= Tr.nr_instance;
		trainauc = calAUC(Tr);

		double testmse = 0;
		double testllh = 0.0;
		double testauc = 0.0;
		#pragma omp parallel for schedule(static) reduction(+: testmse, testllh)
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
			Va.F[i] = 1.0 / (1 + exp(-Fi));

			testmse += (yi - Va.F[i]) * (yi - Va.F[i]);
			testllh += yi * Fi - log(1 + exp(Fi));
		}
		testllh /= Va.nr_instance;
		testmse /= Va.nr_instance;

		testauc = calAUC(Va);
		cout << iter << "\t" << timer.toc() << "\t" << trainllk << "\t" << testllh << "\t" << trainauc << "\t" << testauc << "\t" << trainmse << "\t" << testmse << endl;
	}

	writeWeightFile(Tr);
return 0;
}
