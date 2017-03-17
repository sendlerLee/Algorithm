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

map<int, map<int, double> > trainDataFeature;
map<int, int> trainDataLabel;
map<int, double> trainInstanceWeight;
map<int, double> trainDataPredictVal;
map<int, double> trainInstanceFX;

map<int, string> userid2NameMap;

map<int, map<int, double> > testDataFeature;
map<int, int> testDataLabel;
map<int, double> testInstanceWeight;
map<int, double> testDataPredictVal;
map<int, double> testInstanceFX;

map<int, double> instance2residual;

map<int, double> g_featureDict;
map<int, map<int, double> > model2featureDict;

struct Option
{
    Option() : nr_model(1), nr_iter(100), nr_lr(0.002), nr_reg(0.02) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_model, nr_iter;
    double nr_lr, nr_reg;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: logitboost [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-m <model>: set the number of model\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-i <nr_reg>: set the reg\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    int const argc = static_cast<int>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    int i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-m") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_model = stoi(args[++i]);
        }
	else if(args[i].compare("-i") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_iter = std::stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_lr = std::stof(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_reg = std::stof(args[++i]);
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

void writeWeightFile(map<int, map<int, double> >& model2featureDict)
{
	map<int, double> weightmap;
	for (map<int, map<int, double> >::iterator it = model2featureDict.begin(); it != model2featureDict.end(); ++it) {
		int modelnum = it->first;
		map<int, double> featuremap = it->second;
		for (map<int, double>::iterator fit = featuremap.begin(); fit != featuremap.end(); ++fit) {
			int fid = fit->first;
			double fval = fit->second;
			weightmap[fid] += fval;
		}
	}

	ofstream outfile("feature_weight");
	for (map<int, double>::iterator it = weightmap.begin(); it != weightmap.end(); ++it)
	{
		int fid = it->first;
		double fval = it->second;
		outfile << fid << " " << fval << endl;
        }
	outfile.close();

}

double calAUC(map<int, double>& instancefx, map<int, int>& datalabel)
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
	writeWeightFile(model2featureDict);
	return auc;
}

void loadInstance(ifstream& inputfile, map<int, map<int, double> >& datafeature, map<int, int>& datalabel) 
{    
    string line;
    int idx = 0;
    while(getline(inputfile, line)) {
        istringstream iss(line);
	string userid;
        int target;
        iss >> userid >> target;

        map<int, double> featuredict;
        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_first_of(":", 0);
                int fid = atoi(feature.substr(0, findex).c_str());
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
                featuredict[fid] = fval;
                g_featureDict[fid] = 0;
        }

        datafeature[idx] = featuredict;
        datalabel[idx] = target;
	userid2NameMap[idx] = userid;
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

	loadInstance(trainfile, trainDataFeature, trainDataLabel);
	loadInstance(testfile, testDataFeature, testDataLabel);
	int numTrainInstance = trainDataLabel.size();
	int numTestInstance = testDataLabel.size();

	cout << "Load Data Finish, numTrainInstance:" << numTrainInstance << " numTrainFeature: " << g_featureDict.size() <<  endl;
	cout << "Load Data Finish, numTestInstance:" << numTestInstance << " numTrainFeature: " << g_featureDict.size() <<  endl;

	for (int mi = 0; mi < opt.nr_model; mi += 1) {
		for (map<int, double>::iterator fit = g_featureDict.begin(); fit != g_featureDict.end(); ++fit) {
                	int fid = fit->first;
			model2featureDict[mi][fid] = 0.0;
		}
	}

	for (map<int, int>::iterator tit = trainDataLabel.begin(); tit != trainDataLabel.end(); ++tit) {
		int userid = tit->first;
		int label = tit->second;
		trainInstanceWeight[userid] = 1.0 / numTrainInstance;
		trainInstanceFX[userid] = 0.0;
		trainDataPredictVal[userid] = 0.5;
	}
	for (map<int, int>::iterator tit = testDataLabel.begin(); tit != testDataLabel.end(); ++tit) {
		int userid = tit->first;
		int label = tit->second;
		testInstanceWeight[userid] = 1.0 / numTestInstance;
		testInstanceFX[userid] = 0.0;
		testDataPredictVal[userid] = 0.5;
	}

	int zmax = 2;
	for (int mi = 0; mi < opt.nr_model; mi += 1) {
        	cout << endl << "Greed Add Model: " << mi << "......" << endl << endl;

		map<int, double> userid2wi;
		map<int, double> userid2zi;
        	for (map<int, int>::iterator tit = trainDataLabel.begin(); tit != trainDataLabel.end(); ++tit) {
            		int userid = tit->first;
            		int label = tit->second;

			double yi = label;
			double pi = trainDataPredictVal[userid];
			double zi = (yi - pi) / (pi * (1 - pi));

			if (label == 1) zi = 1.0 / pi;
			if (zi > zmax) zi = zmax;
			if (label == 0) zi = -1.0 / (1 - pi);
			if (zi < 0 - zmax) zi = 0 - zmax;

			double wi = pi * (1 - pi);
        		//wi = (wi <= 1e-12)? 0 : wi;

			userid2wi[userid] = wi;
			userid2zi[userid] = zi;
		}

  		for (int iter = 0; iter < opt.nr_iter; iter += 1) {
			double sumerror = 0.0;
			double sumright = 0.0;
			int cnt = 0;

        		for (map<int, int>::iterator tit = trainDataLabel.begin(); tit != trainDataLabel.end(); ++tit) {
            			int userid = tit->first;
            			int label = tit->second;

				double wi = userid2wi[userid];
				double zi = userid2zi[userid];

				double sumscore = 0.0;
            			map<int, double>& featuredict = trainDataFeature[userid];
            			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
                			int fid = fit->first;
                			double fval = fit->second;
                			sumscore += model2featureDict[mi][fid] * fval;
            			}

				cnt += 1;
				double error = wi * (zi - sumscore) * (zi - sumscore);
				sumerror += error * error;

				double gradientcommon = 0 - wi * (zi - sumscore);
	
            			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
                			int fid = fit->first;
                			double fval = fit->second;
					double gradient = gradientcommon * fval;

					model2featureDict[mi][fid] -= opt.nr_lr * (gradient + opt.nr_reg * model2featureDict[mi][fid]);
				}
			}
        		//cout << "Model: " << mi << " Iterator: " << iter << " Cnt: " << cnt << " SumError: " << sumerror / cnt << endl;
		}

		double trainloss = 0.0;
		double sumposright = 0.0;
		int poscnt = 0;
		double sumnegright = 0.0;
		int negcnt = 0;
		for (map<int, int>::iterator tit = trainDataLabel.begin(); tit != trainDataLabel.end(); ++tit) {
			int userid = tit->first;
			int label = tit->second;

			double sumscore = 0.0;
			map<int, double>& featuredict = trainDataFeature[userid];
			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
				int fid = fit->first;
				double fval = fit->second;
				sumscore += model2featureDict[mi][fid] * fval;
			}

			double fx = trainInstanceFX[userid] + 0.5 * (sumscore);
			trainInstanceFX[userid] = fx;
			trainDataPredictVal[userid] = 1.0 / ( 1.0 + exp(0 - 2 * fx));
			int y = -1;
			if (label == 1) y = 1;
			trainloss += log(1+exp(- y * trainInstanceFX[userid])); 

			if (label == 1) {
				poscnt += 1;
				if (fx >= 0) {
					sumposright += 1;
				}
			}
			if (label == -1) {
				negcnt += 1;
				if (fx < 0) {
					sumnegright += 1;
				}
			}
		}

		double pp = sumposright / poscnt;
		double np = sumnegright / negcnt;
		double f1 = 2 * pp * np / (pp + np);
		cout << endl << "Model: " << mi << " Train Postive Precision : " << sumposright / poscnt << " Negative Precision : " << sumnegright / negcnt << endl;
		cout << "F1 Score: " << f1 << " Precision: " << (sumposright + sumnegright) / (poscnt + negcnt) << " RightCnt: " << sumposright + sumnegright << endl << endl;
		double train_auc = calAUC(trainInstanceFX, trainDataLabel);
		cout << "Train AUC: " << train_auc << endl << endl;

		double testloss = 0.0;
		sumposright = 0.0;
		poscnt = 0;
		sumnegright = 0.0;
		negcnt = 0;
		for (map<int, int>::iterator tit = testDataLabel.begin(); tit != testDataLabel.end(); ++tit) {
			int userid = tit->first;
			int label = tit->second;

			double sumscore = 0.0;
			map<int, double>& featuredict = testDataFeature[userid];
			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
				int fid = fit->first;
				double fval = fit->second;
				sumscore += model2featureDict[mi][fid] * fval;
			}

			double fx = testInstanceFX[userid] + 0.5 * (sumscore);
			testInstanceFX[userid] = fx;
			testDataPredictVal[userid] = 1.0 / ( 1.0 + exp(0 - 2 * fx));

			int y = -1;
			if (label == 1) y = 1;
			testloss += log(1+exp(- y * testInstanceFX[userid])); 

			if (label == 1) {
				poscnt += 1;
				if (fx >= 0) {
					sumposright += 1;
				}
			}
			if (label == -1) {
				negcnt += 1;
				if (fx < 0) {
					sumnegright += 1;
				}
			}
		}
		double auc = calAUC(testInstanceFX, testDataLabel);
		pp = sumposright / poscnt;
		np = sumnegright / negcnt;
		f1 = 2 * pp * np / (pp + np);
		cout << endl << "Model: " << mi << " Test Postive Precision : " << sumposright / poscnt << " Negative Precision : " << sumnegright / negcnt << endl;
		cout << "F1 Score: " << f1 << " Precision: " << (sumposright + sumnegright) / (poscnt + negcnt) << " RightCnt: " << sumposright + sumnegright << endl;
		cout << "AUC: " << auc << endl << endl;
		cout << "Model: " << mi << " Train Loss: " << trainloss * 1.0 / trainDataLabel.size() << endl;
		cout << "Model: " << mi << " Test Loss: " << testloss * 1.0 / testDataLabel.size() << endl << endl;
	}

return 0;
}
