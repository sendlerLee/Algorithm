#include <iostream>
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
#include <string.h>

using namespace std;
// VIDEO TRAVEL GAME NOVEL

//ifstream trainfile("campaign_ext_click.train");
//ifstream testfile("campaign_ext_click.test");

//ifstream trainfile("campaign_ext_conv.train");
//ifstream testfile("campaign_ext_conv.test");


map<int, map<int, double> > trainDataFeature;
map<int, int> trainDataLabel;
map<int, double> trainInstanceWeight;
map<int, double> trainDataPredictVal;
map<int, double> trainInstanceFX;

map<int, map<int, double> > testDataFeature;
map<int, int> testDataLabel;
map<int, double> testInstanceWeight;
map<int, double> testDataPredictVal;
map<int, double> testInstanceFX;

map<int, double> instance2residual;

map<int, double> featureDict;
map<int, map<int, double> > model2featureDict;

int numModel = 1;

//double learning_rate = 0.02;
//double reg = 0.04;

double learning_rate = 0.002;
double reg = 0.02;

char feature_weight[128] = {0};

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

	ofstream outfile(feature_weight);
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
    srand((unsigned)time(NULL));
    string line;
    int linecnt = 0;
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
                featureDict[fid] = rand() % 2000 / 1000.0 - 1;
        }

        datafeature[linecnt] = featuredict;
        datalabel[linecnt] = target;
        linecnt += 1;
    }
}


int main(int argc, char *argv[]) 
{
	if (argc < 4){
		cout<<"Error"<<endl<<argv[0]<<" campaign_ext.train  campaign_ext.test feature_weight(out)"<<endl;
		return -1;
	}

	ifstream trainfile(argv[1]);
	ifstream testfile(argv[2]);
	strcpy(feature_weight, argv[3]);

	loadInstance(trainfile, trainDataFeature, trainDataLabel);
	loadInstance(testfile, testDataFeature, testDataLabel);
	int numTrainInstance = trainDataLabel.size();
	int numTestInstance = testDataLabel.size();

	cout << "Load Data Finish, numTrainInstance:" << numTrainInstance << " numTrainFeature: " << featureDict.size() <<  endl;
	cout << "Load Data Finish, numTestInstance:" << numTestInstance << " numTrainFeature: " << featureDict.size() <<  endl;

	for (int mi = 0; mi < numModel; mi += 1) {
		for (map<int, double>::iterator fit = featureDict.begin(); fit != featureDict.end(); ++fit) {
                	int fid = fit->first;
			//model2featureDict[mi][fid] = rand() % 2000 / 1000.0 - 1;
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
	for (int mi = 0; mi < numModel; mi += 1) {
        	cout << endl << "Greed Add Model: " << mi << "......" << endl << endl;
		int MaxIter = 100;
  		for (int iter = 0; iter < MaxIter; iter += 1) {
        		double sumerror = 0.0;
			double sumright = 0.0;
        		int cnt = 0;
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
				//cout << userid << "\t" << error << "\t" << sumerror << endl;

				double gradientcommon = 0 - wi * (zi - sumscore);
	
            			for (map<int, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
                			int fid = fit->first;
                			double fval = fit->second; 
					double gradient = gradientcommon * fval;

					model2featureDict[mi][fid] -= learning_rate * (gradient + reg * model2featureDict[mi][fid]);
				}
			}
        		cout << "Model: " << mi << " Iterator: " << iter << " Cnt: " << cnt << " SumError: " << sumerror / cnt << endl;
		}

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

			if (label == 1) {
				poscnt += 1;
				if (fx >= 0) {
					sumposright += 1;
				}
			}
			if (label == 0) {
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

			if (label == 1) {
				poscnt += 1;
				if (fx >= 0) {
					sumposright += 1;
				}
			}
			if (label == 0) {
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
	}

return 0;
}
