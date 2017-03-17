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

vector<map<string, double> > trainPosDataFeature;
vector<map<string, double> > trainNegDataFeature;
map<int, string> userid2TrainNameMap;

vector<map<string, double> > testDataFeature;
map<int, int> testDataLabel;
map<int, string> userid2TestNameMap;

map<string, double> g_featweightmap;
int g_pos_cnt = 0;

struct Option
{
    Option() : nr_iter(100), nr_lr(0.0001), nr_reg(0.0001), nr_sample(2), feat_threshold(0.01) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter, nr_sample;
    double nr_lr, nr_reg;
    double feat_threshold;
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
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_sample = stof(args[++i]);
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
	else if(args[i].compare("-h") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.feat_threshold = stof(args[++i]);
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

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile()
{
	ofstream outfile("feature_weight_plr_l1");
	for (map<string, double>::iterator it = g_featweightmap.begin(); it != g_featweightmap.end(); ++it)
	{
		string fid = it->first;
		double fval = it->second;
		outfile << fid << " " << fval << endl;
        }
	outfile.close();
}

double calAUC(vector<pair<int, double> >& instancefxvec, int poscnt, map<int, int>& datalabel, map<int, string>& userid2NameMap)
{
	int negcnt = instancefxvec.size() - poscnt;
	sort(instancefxvec.begin(), instancefxvec.end(), mysortfunc);

        ofstream outfile(opt.Va_out_path);

	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < instancefxvec.size(); ++i) 
	{
        	int userid = instancefxvec[i].first;
		string username = userid2NameMap[userid];
                double fscore = instancefxvec[i].second;
                outfile << username << "\t" << fscore << endl;
		int label = datalabel[userid];
		if (label == 1) 
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
	pvcnt.push_back(instancefxvec.size());
	covcnt.push_back(predictposcnt);
	for (int i = 0; i < pvcnt.size(); i += 1)
		cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	writeWeightFile();
	return auc;
}

void loadTrainInstance(ifstream& inputfile) 
{    
    string line;
    int idx = 0;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

        map<string, double> featuredict;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string fid = feature.substr(0, findex).c_str();
		if (featuredict.count(fid) != 0) continue;
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
                featuredict[fid] = fval;
                g_featweightmap[fid] = 0;
        }

	if (target == 1) {
		trainPosDataFeature.push_back(featuredict);
	} else {
		trainNegDataFeature.push_back(featuredict);
	}
	userid2TrainNameMap[idx] = userid;
        idx += 1;
    }
}

void loadTestInstance(ifstream& inputfile) 
{    
    string line;
    int idx = 0;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;
	if (target != 1) target = 0;

	if (target == 1) g_pos_cnt += 1;

        map<string, double> featuredict;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string fid = feature.substr(0, findex).c_str();
		if (featuredict.count(fid) != 0) continue;
		if (g_featweightmap.count(fid) == 0) continue;
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
                featuredict[fid] = fval;
        }

	testDataFeature.push_back(featuredict);
        testDataLabel[idx] = target;
	userid2TestNameMap[idx] = userid;
        idx += 1;
    }
}

std::vector<std::string> argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

int myShuffle(int* a,int len)
{
	for (int i = len - 1; i > 0; i--)
	{
		srand(i * 1000 + time(NULL));
		int index =  rand() % len;
		int c;
		c = a[index];
		a[index] = a[i];
        	a[i] = c;
	}
	return 0;
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

	ifstream trainfile(opt.Tr_path);
	ifstream testfile(opt.Va_path);

	loadTrainInstance(trainfile);
	int numTrainInstance = userid2TrainNameMap.size();
	cout << "Load Data Finish, numTrainInstance:" << numTrainInstance << " numTrainFeature: " << g_featweightmap.size() <<  endl;

	loadTestInstance(testfile);
	int numTestInstance = testDataLabel.size();
	cout << "Load Data Finish, numTestInstance:" << numTestInstance << " numTrainFeature: " << g_featweightmap.size() <<  endl;

	int SampleCnt = opt.nr_sample;

	vector<pair<int, double> > F_Va(testDataFeature.size());
  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		for (map<string, double>::iterator it = g_featweightmap.begin(); it != g_featweightmap.end(); ++it)
			it->second = 0;

        	for (int idx = 0; idx < trainPosDataFeature.size(); ++idx)
		{
			srand(100000 * iter + idx * 1000 + time(NULL));
			int randval = rand();
			srand(randval + time(NULL));
			if (rand() % 3 != 1) continue;

            		map<string, double>& posfeaturedict = trainPosDataFeature[idx];

			double wxi = 0.0;
			for (map<string, double>::iterator fit = posfeaturedict.begin(); fit != posfeaturedict.end(); ++fit) 
			{
				string fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}

			for (int s = 0; s < SampleCnt; s += 1)
			{
				srand(100000 * iter + idx * 1000 + s * 10 + time(NULL));
				int randval = rand();
				srand(randval + time(NULL));
				int negidx = rand() % trainNegDataFeature.size();
				
				double wxj = 0.0;
            			map<string, double>& negfeaturedict = trainNegDataFeature[negidx];
            			map<string, double> unionfeaturedict = posfeaturedict;
				for (map<string, double>::iterator fit = negfeaturedict.begin(); fit != negfeaturedict.end(); ++fit) 
				{
					string fid = fit->first;
					double fval = fit->second;
					wxj += g_featweightmap[fid] * fval;

					if (posfeaturedict.count(fid) > 0) unionfeaturedict[fid] -= fval;
					else unionfeaturedict[fid] = 0 - fval;
				}

				double gradient = - 1.0 / (1 + exp(wxi - wxj));

				for (map<string, double>::iterator fit = unionfeaturedict.begin(); fit != unionfeaturedict.end(); ++fit) 
				{
					string fid = fit->first;
					double fval = fit->second;
					g_featweightmap[fid] -= opt.nr_lr * (gradient * fval + opt.nr_reg * g_featweightmap[fid]);
				}
			}
		}

		int trimcnt = 0;
		cout << "Trim Begin...Trim Feat: ";
        	//#pragma omp parallel for schedule(static) reduction(+: trimcnt)
		for (map<string, double>::iterator it = g_featweightmap.begin(); it != g_featweightmap.end(); ++it)
		{
			if (fabs(it->second) < opt.feat_threshold) 
			{
				it->second = 0;
				trimcnt += 1;
			}
		}
		cout << trimcnt << "....Total Feat: " << g_featweightmap.size() << " Valid: " << g_featweightmap.size() - trimcnt << " Sparsity: " << trimcnt * 1.0 / g_featweightmap.size() << endl << endl;

		vector<pair<int, double> > instancefxvec(testDataFeature.size());
        	#pragma omp parallel for schedule(static)
        	for (int idx = 0; idx < testDataFeature.size(); ++idx)
		{
			double wxi = 0.0;
			map<string, double>& featuredict = testDataFeature[idx];
			for (map<string, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				string fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}
			instancefxvec[idx] = pair<int, double>(idx, wxi);
			F_Va[idx].first = idx;
			F_Va[idx].second += 1.0 / (1.0 + exp(-wxi));
		}

		double auc = calAUC(F_Va, g_pos_cnt, testDataLabel, userid2TestNameMap);
		cout << "Iter: " << iter << "\tAUC: " << auc << endl << endl;
	}

return 0;
}
