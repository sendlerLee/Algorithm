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

struct Option
{
    Option() : alpha(0.5), beta(1), lambda_1(0.01), lambda_2(0.01){}
    std::string Tr_path, Va_path, Va_out_path;
    double alpha, beta, lambda_1, lambda_2;
};

Option opt;
map<string, int> Feat2IDMap;
vector<double> W;
vector<double> Z;
vector<double> N;

vector<double> F;
vector<int> Y;

int tr_instance;
int tr_field;;

int va_instance;

std::string train_help()
{
    return std::string(
"usage: FTRLProximal [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-a <alpha>: set alpha\n"
"-b <beta>: set beta\n"
"-l1 <lambda_1>: L1 Reg lambda_1\n"
"-l2 <lambda_2>: L2 Reg lambda_2\n");
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

void writeWeightFile(vector<double>& W)
{
        ofstream outfile(opt.Va_out_path);
	for (map<string, int>::iterator it = Feat2IDMap.begin(); it != Feat2IDMap.end(); ++it) 
	{
		string feat = it->first;
		int j = Feat2IDMap[feat];
		double w = W[j];
		outfile << feat << " " << w << endl;
        }
	outfile.close();
}

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

double calAUC(vector<double>& F, vector<int>& Y)
{
	int poscnt = 0;
	vector<pair<int, double>> userid2score(F.size());
	map<int, int> userid2label;
	for (int i = 0; i < F.size(); i += 1)
	{
		userid2score[i] = pair<int, double>(i, F[i]);

		int yi = Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[i] = yi;
	}
	int negcnt = F.size() - poscnt;
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
		else 
			belownegcnt -= 1;
	}

	double auc = score * 1.0 / poscnt;
	return auc;
}

int sign(double x)
{
	if(x < 0) 
		return -1;
	else
		return 1;
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
	try 
	{
		opt = parse_option(argv_to_args(argc, argv));
	}
	catch(std::invalid_argument const &e) 
	{	
		std::cout << e.what();
		return EXIT_FAILURE;
	}

	omp_set_num_threads(50);

	string line;

	double alpha = opt.alpha;
	double beta = opt.beta;
	double lambda_1 = opt.lambda_1;
	double lambda_2 = opt.lambda_2;

	srandom(time(NULL));

	int i = 0;
	ifstream trainfile(opt.Tr_path);
	while(getline(trainfile, line)) 
	{
		istringstream iss(line + "\tF0:1");
		string userid;
		int yi;
		iss >> userid >> yi;
		if (yi != 1) yi = 0;

		double Fi = 0.0;
		string feature;
		map<int, double> instfeatdic;	
        	while (iss) 
		{
                	iss >> feature;
                	int findex = feature.find_last_of(":");
                	string feat = feature.substr(0, findex).c_str();
                	double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			int j = -1;

			/*
			if (Feat2IDMap.count(feat) == 0) 
			{
				int j = Feat2IDMap.size();
				Feat2IDMap[feat] = j;

				W.push_back(0);
				N.push_back(0);
				Z.push_back(0);
			}
			*/

			if (Feat2IDMap.count(feat) == 0) 
			{
				if ((rand() % 1000) /1000.0 > 0.1) 
					continue;
				else 
				{
					int j = Feat2IDMap.size();
					Feat2IDMap[feat] = j;

					W.push_back(0);
					N.push_back(0);
					Z.push_back(0);
				}
			}

			j = Feat2IDMap[feat];

			if (instfeatdic.count(j) != 0) continue;

			if (abs(Z[j]) <= lambda_1)
				W[j] = 0.0; 
			else 
				W[j] = - (Z[j] - sign(Z[j]) * lambda_1) / ((beta + sqrt(N[j])) / alpha + lambda_2);

			Fi += W[j] * x;
			instfeatdic[j] = x;
        	}

		double pi = 1.0 / (1 + exp(- Fi));
		for (map<int, double>::iterator it = instfeatdic.begin(); it != instfeatdic.end(); ++it) 
		{
			int j = it->first;
			double x = it->second;
			double g = (pi - yi) * x;
			double tao = (sqrt(N[j] + g * g) - sqrt(N[j])) / alpha;
			Z[j] += g - tao * W[j];
			N[j] += g * g;
		}

		i += 1;
    	}
	tr_instance = i;
    	tr_field = W.size();

	int trimcnt = 0;
	#pragma omp parallel for schedule(static) reduction(+: trimcnt)
	for (int j = 0; j < W.size(); j += 1) 
	{
		if (abs(W[j]) <= 0)
			trimcnt += 1;
	}
	double sparity = trimcnt * 100.0 / tr_field;

	i = 0;
	double va_mse = 0.0;
	double va_logloss = 0.0;
	ifstream vafile(opt.Va_path);
	while(getline(vafile, line))
	{
		istringstream iss(line + "\tF0:1");
		string userid;
		int yi;
		iss >> userid >> yi;
		if (yi != 1) yi = 0;

		double Fi = 0.0;
        	string feature;
		map<int, double> instfeatdic;	
        	while (iss) 
		{
                	iss >> feature;
                	int findex = feature.find_last_of(":");
                	string feat = feature.substr(0, findex).c_str();
			if (Feat2IDMap.count(feat) == 0) continue;

			int j = Feat2IDMap[feat];
			if (instfeatdic.count(j) != 0) continue;

                	double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
			instfeatdic[j] = x;

			Fi += W[j] * x;
        	}
	
		F.push_back(Fi);
		Y.push_back(yi);
		double pi = 1.0 / (1 + exp(-Fi));

		va_mse += (yi - pi) * (yi - pi);

		if (pi > 0.999) pi = 0.999;
		if (pi < 0.001) pi = 0.001;
		va_logloss -= yi * log(pi) + (1 - yi) * log(1 - pi);

		i += 1;
	}

	va_instance = i;
	va_logloss /= va_instance;
	va_mse /= va_instance;

	double vaauc = calAUC(F, Y);
	cout << "Instance\tField" << endl;
	cout << va_instance << "\t" << tr_field << endl;

	cout << "Logloss\tMse\tAUC\tSparity" << endl;
	cout << va_logloss << "\t" << va_mse << "\t" << vaauc << "\t" << sparity << endl;

	writeWeightFile(W);
}
