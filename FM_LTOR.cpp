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
#include <random>
#include <omp.h>
		

using namespace std;

vector<map<string, double> > trainPosDataFeature;
vector<map<string, double> > trainNegDataFeature;
map<int, string> userid2TrainNameMap;

vector<map<string, double> > testDataFeature;
map<int, int> testDataLabel;
map<int, string> userid2TestNameMap;

map<string, vector<int> > userNegInstance;
map<int, string> Idx2UserMap;

double w0;
map<string, double> w1dim;
map<string, vector<double>> wcrossdim;
vector<double> sum_diff;
vector<double> sum_pos;
vector<double> sum_neg;

struct Option
{
    Option() : nr_iter(100), nr_lr(0.0001), nr_reg(0.0001), nr_sample(2), num_factor(8), feat_threshold(0.01) {}
    std::string Tr_path, Va_path, Va_out_path;
    int nr_iter, nr_sample;
    double nr_lr, nr_reg, num_factor;
    double feat_threshold;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: LTOR [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-d <nr_lr>: set the latent variable\n"
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
        else if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.num_factor = stof(args[++i]);
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
	ofstream outfile("feature_weight");
    outfile << "#global bias W0" << endl;   
    outfile << w0 << endl;
    outfile << "#unary interactions Wj" << endl;
	for (map<string, double>::iterator it = w1dim.begin(); it != w1dim.end(); ++it)
	{
		string fid = it->first;
		double fval = it->second;
        if(fval != 0)
		    outfile << fid << " " << fval << endl;
    }
    outfile << "#pairwise interactions Vj,f" << endl; 
    for (map<string, double>::iterator it = w1dim.begin(); it != w1dim.end(); ++it)
    {
        string fid = it->first;
        outfile << fid << ":";
        for(int f = 0; f < opt.num_factor; f++) {
            double fval = wcrossdim[fid][f];
            outfile << fval << " ";
        }
        outfile << endl;
    }

	outfile.close();
}

double calAUC(vector<pair<int, double> >& instancefxvec, int poscnt, map<int, int>& datalabel, map<int, string>& userid2NameMap)
{
	int negcnt = instancefxvec.size() - poscnt;
	sort(instancefxvec.begin(), instancefxvec.end(), mysortfunc);

    //ofstream outfile(opt.Va_out_path);

	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < instancefxvec.size(); ++i) 
	{
        int userid = instancefxvec[i].first;
		string username = userid2NameMap[userid];
        double fscore = instancefxvec[i].second;
		int label = datalabel[userid];
        //outfile << username << "\t" << label << "\t" << fscore << endl;
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
   // outfile.close();
	pvcnt.push_back(instancefxvec.size());
	covcnt.push_back(predictposcnt);
	//for (int i = 0; i < pvcnt.size(); i += 1)
	//	cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	writeWeightFile();
	return auc;
}

void loadTrainInstance(ifstream& inputfile) 
{    
    string line;
    int idx = 0;
    std::default_random_engine e; //引擎  
    std::normal_distribution<double> n(0, 0.01); //均值, 方差
    while(getline(inputfile, line)) {
        istringstream iss(line);
	    string userid;
        int target;
        iss >> target >> userid;
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
            if(w1dim.count(fid) != 0) continue;
            w1dim[fid] = 0;
            wcrossdim[fid].assign(opt.num_factor,0);
            for(int f = 0; f < opt.num_factor; f++) {
                wcrossdim[fid][f] = n(e);
                //std::cout << "wcrossdim[" << fid << "][" << f << "]=" << wcrossdim[fid][f] << endl;
            }
        }


        if (target >= 1) {
            int posidx = trainPosDataFeature.size();
            Idx2UserMap[posidx] = userid;
            trainPosDataFeature.push_back(featuredict);
        } else {
            int negidx = trainNegDataFeature.size();
            userNegInstance[userid].push_back(negidx);
            trainNegDataFeature.push_back(featuredict);
        }

        userid2TrainNameMap[idx] = userid;
        idx += 1;
        if(idx % 10000 == 0) std::cout << ".";
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
        iss >> target >> userid;
	    if (target != 1) target = 0;

        map<string, double> featuredict;
        string feature;
        while (iss) {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string fid = feature.substr(0, findex).c_str();
		    if (featuredict.count(fid) != 0) continue;
		    if (w1dim.count(fid) == 0) continue;
            double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
            featuredict[fid] = fval;
        }

	    testDataFeature.push_back(featuredict);
        testDataLabel[idx] = target;
	    userid2TestNameMap[idx] = userid;
        idx += 1;
        if(idx % 10000 == 0) std::cout << ".";
    }
}

double computeSum_f(map<string, double>& featuredict,vector<double>& sum_f) {
	double result = w0;	
    map<double,vector<double>> resultMap;
	vector<double> sum_sqrt_f(opt.num_factor);
	for (map<string, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit)
	{
		string fid = fit->first;
		double fval = fit->second;
		result += w1dim[fid] * fval;
        //if(w1dim[fid]*fval > 1)
          //  std::cout << "uniay:" << fid << "====" << fval  << "====" << w1dim[fid] << "====" << w1dim[fid]*fval << endl;
	}

    //#pragma omp parallel for schedule(static) reduction(+: result)
	for(int f = 0; f < opt.num_factor; f++) {
		sum_f[f] = 0;
		sum_sqrt_f[f] = 0;
		for(map<string, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) {
			string fid = fit -> first;
			double fval = fit -> second;
			double v = wcrossdim[fid][f] * fval;
            //if(v > 1) 
            //    std::cout << "pair:" << fid << "====" << fval  << "====" << wcrossdim[fid][f] << "====" << v << endl; 
			sum_f[f] += v;
			sum_sqrt_f[f] += v * v;
		}
        //std::cout << sum_f[f] << endl;
		result += 0.5 * (sum_f[f] * sum_f[f] - sum_sqrt_f[f]);
	}
	return result;
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

    //omp_set_num_threads(static_cast<int>(50));

	ifstream trainfile(opt.Tr_path);
	ifstream testfile(opt.Va_path);

	loadTrainInstance(trainfile);
	int numTrainInstance = userid2TrainNameMap.size();
	cout << "Load Data Finish, numTrainInstance:" << numTrainInstance << " numTrainFeature: " << w1dim.size() <<  endl;

	loadTestInstance(testfile);
	int numTestInstance = testDataLabel.size();
	cout << "Load Data Finish, numTestInstance:" << numTestInstance << endl;
	
    sum_diff.assign(opt.num_factor,0);
    sum_pos.assign(opt.num_factor,0);
    sum_neg.assign(opt.num_factor,0);

  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
    {
        for (int idx = 0; idx < trainPosDataFeature.size();  ++idx) 
		{    		
            map<string, double>& posfeaturedict = trainPosDataFeature[idx];
			
            int SampleCnt = opt.nr_sample;
			string userid = Idx2UserMap[idx];
			if (userNegInstance[userid].size() < SampleCnt)
				SampleCnt = userNegInstance[userid].size();

            //std::cout << "sampleCnt:" << userNegInstance[userid].size() << ",userid:" << userid << endl;

            std::random_shuffle (userNegInstance[userid].begin(), userNegInstance[userid].end() );
			for (int s = 0; s < SampleCnt; s += 1)
			{
				int negidx = userNegInstance[userid][s];
				
            	map<string, double>& negfeaturedict = trainNegDataFeature[negidx];
            	map<string, double> unionfeaturedict = posfeaturedict;
                
                double wxi = computeSum_f(posfeaturedict,sum_pos);
                double wxj = computeSum_f(negfeaturedict,sum_neg);

                double gradient = - 1.0 / (1 + exp(wxi - wxj));
                //std::cout << "wxi:" << wxi << ",wxj:" << wxj << ",gradient:" << gradient << endl;
                    
                for (map<string, double>::iterator fit = negfeaturedict.begin(); fit != negfeaturedict.end(); ++fit) 
				{
					string fid = fit->first;
					double fval = fit->second;

					if (posfeaturedict.count(fid) > 0) unionfeaturedict[fid] -= fval;
					else unionfeaturedict[fid] = 0 - fval;
				}
                w0 -= opt.nr_lr * (gradient - opt.nr_reg * w0);
				
                for (map<string, double>::iterator fit = unionfeaturedict.begin(); fit != unionfeaturedict.end(); ++fit) 
				{
					string fid = fit->first;
					double fval = fit->second;
				    w1dim[fid] -= opt.nr_lr * (fval * gradient - opt.nr_reg * w1dim[fid]);
				}

                //#pragma omp parallel for schedule(dynamic)
                for(int f = 0; f < opt.num_factor; f++) {
                    for (map<string, double>::iterator fit = unionfeaturedict.begin(); fit != unionfeaturedict.end(); ++fit)
                    {
                        string fid = fit->first;
                        double fval = fit->second;
                        double grad_pos = 0;
                        double grad_neg = 0;
                        if(posfeaturedict.count(fid) > 0) {
                            fval = posfeaturedict[fid];
                            grad_pos = sum_pos[f] * fval - wcrossdim[fid][f] * fval * fval;
                        }
                        if(negfeaturedict.count(fid) > 0) {
                            fval = negfeaturedict[fid];
                            grad_neg = sum_neg[f] * fval - wcrossdim[fid][f] * fval * fval;
                        }
                        wcrossdim[fid][f] -= opt.nr_lr * (gradient * (grad_pos - grad_neg) - opt.nr_reg * wcrossdim[fid][f]);                          
                       // if(fid.compare("9") == 0)
                         //   std::cout << "wcrossdim[9][" << f << "]:" << wcrossdim[fid][f] << ",sum_pos[" << f << "]:" << sum_pos[f] << ",sum_neg[" << f << "]:" << sum_neg[f] << endl;
                    }
                }
			}
		}


		int poscnt = 0;
		vector<pair<int, double> > instancefxvec(testDataFeature.size());
        ofstream outfile(opt.Va_out_path);
        //#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < testDataFeature.size(); ++idx)
		{
			int yi = testDataLabel[idx];
			if (yi >= 1) poscnt += 1;

			double wxi = 0.0;
			map<string, double>& featuredict = testDataFeature[idx];
            wxi = computeSum_f(featuredict,sum_diff);
			instancefxvec[idx] = pair<int, double>(idx, wxi);
            string userid = userid2TestNameMap[idx];
            outfile << userid << " " << testDataLabel[idx] << " " << wxi << endl;
		}

        outfile.close();
		double auc = calAUC(instancefxvec, poscnt, testDataLabel, userid2TestNameMap);
		cout << "Iter: " << iter << "\tAUC: " << auc << endl;
	}

return 0;
}
