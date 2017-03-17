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

using namespace std;

vector<map<int, double> > trainPosDataFeature;
vector<map<int, double> > trainNegDataFeature;
map<int, string> userid2TrainNameMap;

vector<map<int, double> > testDataFeature;
map<int, int> testDataLabel;
map<int, string> userid2TestNameMap;

map<int, double> g_featweightmap;

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

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

void writeWeightFile()
{
	ofstream outfile("feature_weight");
	for (map<int, double>::iterator it = g_featweightmap.begin(); it != g_featweightmap.end(); ++it)
	{
		int fid = it->first;
		double fval = it->second;
		outfile << fid << " " << fval << endl;
        }
	outfile.close();
}

double calAUC(map<int, double>& instancefx, map<int, int>& datalabel, map<int, string>& userid2NameMap)
{
	int poscnt = 0, negcnt = 0;
	vector<pair<int, double> > instancefxvec;
	for (map<int, double>::iterator it = instancefx.begin(); it != instancefx.end(); ++it) {
		int userid = it->first;
		double score = it->second;
		instancefxvec.push_back(pair<int, double>(userid, score));
		int label = datalabel[userid];
		if (label == 1) poscnt += 1;
	}
	negcnt = instancefx.size() - poscnt;
	sort(instancefxvec.begin(), instancefxvec.end(), mysortfunc);

        ofstream outfile(opt.Va_out_path);

	vector<int> pvcnt;
	vector<int> covcnt;
	int belowposcnt = poscnt;
	int instancesize = instancefxvec.size();
	double score = 0;
	double avgidx = 0;
	int startpoint = 1000;
	int predictposcnt = 0;
	for (size_t i = 0; i < instancefxvec.size(); ++i) {
        	int userid = instancefxvec[i].first;
		string username = userid2NameMap[userid];
                double fscore = instancefxvec[i].second;
                outfile << username << "\t" << fscore << endl;
		int label = datalabel[userid];
		if (label == 1) {
			score += (instancesize - i - belowposcnt) * 1.0 / (poscnt);
			belowposcnt -= 1;
			avgidx += i * 1.0 / poscnt;
			predictposcnt += 1;
		}
		if (i == startpoint) {
			pvcnt.push_back(i);
			covcnt.push_back(predictposcnt);
			startpoint *= 2;
		}
	}
        outfile.close();
	pvcnt.push_back(instancefxvec.size());
	covcnt.push_back(predictposcnt);
	cout << "avgidx: " << avgidx * 1.0 / poscnt << " " << instancesize << " IDX: " << avgidx * 1.0 / (instancesize) << endl;
	for (int i = 0; i < pvcnt.size(); i += 1)
		cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / (negcnt);
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

        map<int, double> featuredict;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_first_of(":", 0);
                int fid = atoi(feature.substr(0, findex).c_str());
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

        map<int, double> featuredict;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_first_of(":", 0);
                int fid = atoi(feature.substr(0, findex).c_str());
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

	int SampleCnt = 2;

  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		int negidx = 0;
        	for (int idx = 0; idx < trainPosDataFeature.size();  ++idx) 
		{
            		map<int, double>& posfeaturedict = trainPosDataFeature[idx];

			double wxi = 0.0;
			for (map<int, double>::iterator fit = posfeaturedict.begin(); fit != posfeaturedict.end(); ++fit) 
			{
				int fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}

			for (int s = 0; s < SampleCnt; s += 1)
			{
				//negidx += s;
				//negidx = negidx % trainNegDataFeature.size();

				srand(100000 * s + 10000 * iter + idx);
				int negidx = rand() % trainNegDataFeature.size();
				//cout << iter << " " << idx << " " << negidx << endl;
				
				double wxj = 0.0;
            			map<int, double>& negfeaturedict = trainNegDataFeature[negidx];
            			map<int, double> unionfeaturedict = posfeaturedict;
				for (map<int, double>::iterator fit = negfeaturedict.begin(); fit != negfeaturedict.end(); ++fit) 
				{
					int fid = fit->first;
					double fval = fit->second;
					wxj += g_featweightmap[fid] * fval;

					if (posfeaturedict.count(fid) > 0) unionfeaturedict[fid] -= fval;
					else unionfeaturedict[fid] = 0 - fval;
				}

				double gradient = - 1.0 / (1 + exp(wxi - wxj));

				for (map<int, double>::iterator fit = unionfeaturedict.begin(); fit != unionfeaturedict.end(); ++fit) 
				{
					int fid = fit->first;
					double fval = fit->second;
					g_featweightmap[fid] -= opt.nr_lr * (gradient * fval + opt.nr_reg * g_featweightmap[fid]);
					//g_featweightmap[fid] -= opt.nr_lr * (gradient * fval - opt.nr_reg * g_featweightmap[fid]);
				}
			}
		}

		double llh = 0.0;
        	for (int idx = 0; idx < trainPosDataFeature.size(); ++idx)
		{
			double wxi = 0.0;
            		map<int, double>& featuredict = trainPosDataFeature[idx];
			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				int fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}
			double pi = 1 / (1 + exp(- wxi));
			llh += log(pi);
		}
        	for (int idx = 0; idx < trainNegDataFeature.size(); ++idx)
		{
			double wxi = 0.0;
            		map<int, double>& featuredict = trainNegDataFeature[idx];
			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				int fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}
			double pi = 1 / (1 + exp(- wxi));
			llh += log(1 - pi);
		}
		llh /= userid2TrainNameMap.size();

		double testllh = 0.0;
		map<int, double> instancefx;
        	for (int idx = 0; idx < testDataFeature.size(); ++idx)
		{
			int yi = testDataLabel[idx];

			double wxi = 0.0;
			map<int, double>& featuredict = testDataFeature[idx];
			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				int fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}
			double pi = 1 / (1 + exp(- wxi));
			instancefx[idx] = pi;

			testllh += yi * log(pi) + (1 - yi) * log(1 - pi);
		}
		testllh /= testDataLabel.size();

		double auc = calAUC(instancefx, testDataLabel, userid2TestNameMap);
		cout << "Iter: " << iter << "\tTrain llh: " << llh << "\tTest llh: " << testllh << "\tAUC: " << auc << endl;

		//cout << "Iter: " << iter << "\tTrain llh: " << llh << "\tTest llh: " << testllh << endl;
	}
	//double auc = calAUC(instancefx, testDataLabel, userid2TestNameMap);
	//cout << "Test AUC: " << auc << endl << endl;

return 0;
}
