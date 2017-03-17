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
#include <thread>
#include <omp.h>

using namespace std;

vector<map<int, double> > TrPosF;
vector<map<int, double> > TrNegF;

vector<map<int, double> > VaF;
vector<int> VaY;

map<string, int> Feat2IDMap;

vector<double> W;

int nr_field;
int nr_instance;

struct Option
{
    Option() : nr_iter(100), nr_lr(0.0001), lambda_2(0.0001), nr_sample(2), lambda_1(0.01) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter, nr_sample;
    double nr_lr;
    double lambda_1, lambda_2;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: PLR [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-s <nr_sample>: set the sample count\n"
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
            opt.nr_lr = stod(args[++i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_2 = stod(args[++i]);
        }
	else if(args[i].compare("-l1") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.lambda_1 = stod(args[++i]);
        }
        else
            break;
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile()
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

void loadTrainInstance(ifstream& inputfile) 
{    
	string line;
	while(getline(inputfile, line)) 
	{
		istringstream iss(line);
		string userid;
		int yi;
		iss >> userid >> yi;
		if (yi != 1) yi = 0;

		map<int, double> featuredict;
		string feature;
		while (iss) 
		{
			iss >> feature;
			int findex = feature.find_last_of(":");
			string feat = feature.substr(0, findex).c_str();

			int j = -1;
			if (Feat2IDMap.count(feat) == 0) 
			{
				int j = Feat2IDMap.size();
				Feat2IDMap[feat] = j;

				W.push_back(0.0);
			}
			j = Feat2IDMap[feat];

			if (featuredict.count(j) != 0) continue;

			double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
			featuredict[j] = x;
		}

		if (yi == 1) 
			TrPosF.push_back(featuredict);
		else
			TrNegF.push_back(featuredict);

		nr_instance += 1;
	}

	TrPosF.shrink_to_fit();
	TrNegF.shrink_to_fit();
}

void loadTestInstance(ifstream& inputfile) 
{    
	string line;
	while(getline(inputfile, line)) 
	{
		istringstream iss(line);
		string userid;
		int yi;
		iss >> userid >> yi;
		if (yi != 1) yi = 0;

		map<int, double> featuredict;
		string feature;
		while (iss) 
		{
			iss >> feature;
			int findex = feature.find_last_of(":");
			string feat = feature.substr(0, findex).c_str();
			if (Feat2IDMap.count(feat) == 0) continue;

			int j = Feat2IDMap[feat];
			if (featuredict.count(j) != 0) continue;

			double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
			featuredict[j] = x;
        	}
		if (featuredict.size() == 0) continue;

		VaF.push_back(featuredict);
		VaY.push_back(yi);
	}
	VaF.shrink_to_fit();
	VaY.shrink_to_fit();
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

double getPLRFeat() 
{
	std::ofstream trainfeatfile(opt.Tr_path + ".plrfeat");

	for (int i = 0; i < TrPosF.size(); ++i)
	{
		double wxi = 0.0;
		map<int, double>& pfdict = TrPosF[i];
		for (map<int, double>::iterator it = pfdict.begin(); it != pfdict.end(); ++it) 
		{
			int j = it->first;
			double x = it->second;
			wxi += W[j] * x;
		}
		trainfeatfile << "trpos_" << i << "\t1\tx:" << wxi << "\tx2:" << sign(wxi) * log(abs(wxi) + 1) << "\tx5:1" << endl;
	}

	for (int i = 0; i < TrNegF.size(); ++i)
	{
		double wxi = 0.0;
		map<int, double>& nfdict = TrNegF[i];
		for (map<int, double>::iterator it = nfdict.begin(); it != nfdict.end(); ++it) 
		{
			int j = it->first;
			double x = it->second;
			wxi += W[j] * x;
		}
		trainfeatfile << "trneg_" << i << "\t0\tx:" << wxi << "\tx2:" << sign(wxi) * log(abs(wxi) + 1) << "\tx5:1" << endl;
	}

	std::ofstream testfeatfile(opt.Va_path + ".plrfeat");
	for (int i = 0; i < VaF.size(); ++i)
	{
		double wxi = 0.0;
		map<int, double>& fdict = VaF[i];
		for (map<int, double>::iterator it = fdict.begin(); it != fdict.end(); ++it) 
		{
			int j = it->first;
			double x = it->second;
			wxi += W[j] * x;
		}
		testfeatfile << "Va_" << i << "\t" << VaY[i] << "\tx:" << wxi << "\tx2:" << sign(wxi) * log(abs(wxi) + 1) << "\tx5:1" << endl;
	}

    return 0;
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

	ifstream trainfile(opt.Tr_path);
	ifstream testfile(opt.Va_path);

	loadTrainInstance(trainfile);
	nr_field = W.size();
	cout << "Load Data Finish, numTrainInstance:" << nr_instance << " numTrainFeature: " << nr_field <<  endl;

	loadTestInstance(testfile);
	int numTestInstance = VaY.size();
	cout << "Load Data Finish, numTestInstance:" << numTestInstance << endl;

        omp_set_num_threads(20);               

	cout << "Iter\tAUC\tSparity" << endl;
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		srandom(time(NULL));
        	for (int idx = 0; idx < TrPosF.size(); ++idx)
		{
            		map<int, double>& pfdict = TrPosF[idx];
			double wxi = 0.0;
			for (map<int, double>::iterator it = pfdict.begin(); it != pfdict.end(); ++it) 
			{
				int j = it->first;
				double x = it->second;
				wxi += W[j] * x;
			}

			for (int s = 0; s < opt.nr_sample; s += 1)
			{
				int randval = rand();
				int negidx = rand() % TrNegF.size();
				
				double wxj = 0.0;
            			map<int, double> ufdict = pfdict;
            			map<int, double>& nfdict = TrNegF[negidx];
				for (map<int, double>::iterator it = nfdict.begin(); it != nfdict.end(); ++it) 
				{
					int j = it->first;
					double x = it->second;
					wxj += W[j] * x;

					if (pfdict.count(j) > 0) ufdict[j] -= x;
					else ufdict[j] = 0 - x;
				}

				double gradient = - 1.0 / (1 + exp(wxi - wxj));

				for (map<int, double>::iterator it = ufdict.begin(); it != ufdict.end(); ++it) 
				{
					int j = it->first;
					double x = it->second;
					W[j] -= opt.nr_lr * (gradient * x + opt.lambda_2 * W[j]);
				}
			}
		}

		int trimcnt = 0;
        	#pragma omp parallel for schedule(static) reduction(+: trimcnt)
		for (int j = 0; j < W.size(); j += 1)
		{
			if (fabs(W[j]) < opt.lambda_1) 
			{
				W[j] = 0;
				trimcnt += 1;
			}
		}
		double sparity = trimcnt * 100.0 / nr_field;

		vector<double> F(VaF.size(), 0);
		vector<int> Y(VaF.size(), 0);
        	#pragma omp parallel for schedule(static)
        	for (int i = 0; i < VaF.size(); ++i)
		{
			double wxi = 0.0;
			map<int, double>& fdict = VaF[i];
			for (map<int, double>::iterator it = fdict.begin(); it != fdict.end(); ++it) 
			{
				int j = it->first;
				double x = it->second;
				wxi += W[j] * x;
			}
			F[i] = wxi;
			Y[i] = VaY[i];
		}

		double auc = calAUC(F, Y);
		cout << iter << "\t" << auc << "\t" << sparity << endl;
	}
 
	writeWeightFile();
	getPLRFeat();
return 0;
}
