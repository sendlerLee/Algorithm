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
#include<boost/unordered_map.hpp>

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
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(begin-begin);
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


struct Option
{
    Option() : alpha(0.5), beta(1), lambda_1(0.01), lambda_2(0.01), algorithm(3), nr_iter(10), ndim(8), rank(0), cls(1), pcnt(1.5) {}
    std::string Tr_path, Va_path, FeatWeight_path;
    float alpha, beta, lambda_1, lambda_2;
    float pcnt;
    int rank, cls;
    int algorithm;
    int nr_iter;
    int ndim;
};

Option opt;

struct sort_by_v_ascend
{
    bool operator() (std::pair<uint32_t, float> const lhs, std::pair<uint32_t, float> const rhs)
    {
        return lhs.second < rhs.second;
    }
};

double medianw(std::vector<float> const &Y, std::vector<float> const &W)
{
	vector<std::pair<uint32_t, float>> instscorevec(Y.size());
	for(uint32_t i = 0; i < Y.size(); ++i)
		instscorevec[i] = pair<uint32_t, float>(i, Y[i]);
	std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v_ascend());
	double halfsum = 0.5 * std::accumulate(W.begin(), W.end(), 0.0);
	double tempsum = 0.0;
	for(int i = 0; i < instscorevec.size(); i += 1)
	{
		uint32_t idx = instscorevec[i].first;
		double m = instscorevec[i].second;
		double w = W[idx];
		tempsum += w;
		if (tempsum >= halfsum) return m;
	}
}

struct Problem
{
    Problem() {}

    int nr_instance, nr_field;

    boost::unordered_map<std::string, int> feat2idmap;
    boost::unordered_map<int, std::string> id2instance;

    boost::unordered_map<int, std::string> id2qid;
    boost::unordered_map<std::string, vector<int>> qid2posInstMap;
    boost::unordered_map<std::string, vector<int>> qid2InstMap;

    std::map<int, int> fid2cnt;

    std::vector<vector<pair<int, float>>> X;

    std::vector<float> Y;
    std::vector<float> F;

    std::vector<float> W;
    std::vector<float> N;
    std::vector<float> Z;

    std::vector<vector<float>> V;
    std::vector<vector<float>> VN;
    std::vector<vector<float>> VZ;

    float F0;
};


std::string train_help()
{
    return std::string(
"usage: FTRLProximal [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-alg <algorithm>: 1: Gaussian; 2: AdaBoost; 3: Bernoulli; 4: Poisson; 5: Laplace; 6:MAPE; default 3 \n"
"-r <rank>: set rank\n"
"-c <classification>: set classification\n"
"-p <pcnt>: set pcnt\n"
"-a <alpha>: set alpha\n"
"-b <beta>: set beta\n"
"-d <Dim>: set latent factor dim\n"
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
	   if(args[i].compare("-i") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_iter = stoi(args[++i]);
        }
	   else if(args[i].compare("-alg") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.algorithm = stoi(args[++i]);
        }
	   else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.rank = stoi(args[++i]);
        }
	   else if(args[i].compare("-c") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.cls = stoi(args[++i]);
        }
	   else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.pcnt = stod(args[++i]);
        }

        else if(args[i].compare("-a") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.alpha = stod(args[++i]);
        }
	   else if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.ndim = stoi(args[++i]);
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
        else break;
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command2");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.FeatWeight_path = args[i++];

    return opt;
}


void writeWeightFile(Problem& Tr, Problem& Va, int iter)
{
    ofstream outfile(opt.FeatWeight_path + "." + to_string(iter));
	for (boost::unordered_map<string, int>::iterator it = Tr.feat2idmap.begin(); it != Tr.feat2idmap.end(); ++it)
	{
		string feat = it->first;
		int j = it->second;
		float w = Tr.W[j];
		outfile << feat << " " << w << " ";
		for (int k = 0; k < opt.ndim - 1; k += 1)
			outfile << Tr.V[j][k] << " ";
		if (opt.ndim - 1 >= 0)
			outfile << Tr.V[j][opt.ndim - 1] << endl;
		
    }
	outfile.close();

	ofstream va_res_outfile(opt.Va_path + "." + to_string(iter) + ".pred");
	for (int i = 0; i < Tr.nr_instance; i += 1) 
	{
		string inst = Tr.id2instance[i];
		float pred = Tr.F[i];
		va_res_outfile << inst << "\ttrain\t" << Tr.Y[i] << "\t" << pred << endl;
    }

	for (int i = 0; i < Va.nr_instance; i += 1) 
	{
		string inst = Va.id2instance[i];
		float pred = Va.F[i];
		va_res_outfile << inst << "\tvalidation\t" << Va.Y[i] << "\t" << pred << endl;
    }
	va_res_outfile.close();
}

int sign(float x)
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


void loadTrainInstance(Problem& Tr, ifstream& inputfile) 
{
    string line;
    while(getline(inputfile, line)) 
    {
		istringstream iss(line);
        string userid;
        float target;
		iss >> target >> userid;

        //if (opt.algorithm == 2 || opt.algorithm == 3) 
		if (target != 1) target = 0;

        Tr.Y.push_back(target);

		vector<pair<string, float>> feat2val;
		feat2val.push_back(pair<string, float>("F0", 1.0));

        string feature;
        while (! iss.eof()) 
        {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string feat = feature.substr(0, findex).c_str();
            float x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
            if (x == 0.0) continue;
            feat2val.push_back(pair<string, float>(feat, x));
        }

        Tr.X.push_back(std::vector<pair<int, float>>());
        int i = Tr.X.size() - 1;
        Tr.X[i].resize(feat2val.size(), pair<int, float>());
		Tr.id2instance[i] = userid;

		string qid = userid//.substr(0, userid.find_first_of("_")).c_str();
		Tr.id2qid[i] = qid;
		if (target > 0) Tr.qid2posInstMap[qid].push_back(i);
		Tr.qid2InstMap[qid].push_back(i);

        int j = 0;
        for (vector<pair<string, float>>::iterator it = feat2val.begin(); it != feat2val.end(); ++it) 
        {
            string feat = it->first;
            float x = it->second;

			int f = -1;
			if (Tr.feat2idmap.find(feat) == Tr.feat2idmap.end())
			{
				f = Tr.feat2idmap.size();
				Tr.feat2idmap[feat] = f;
			}
			f = Tr.feat2idmap[feat];
			Tr.fid2cnt[f] += 1;

            Tr.X[i][j] = pair<int, float>(f, x);
            j += 1;
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
        float target;
		iss >> target >> userid;

        //if (opt.algorithm == 2 || opt.algorithm == 3) 
		if (target != 1) target = 0;

		vector<pair<string, float>> feat2val;
		feat2val.push_back(pair<string, float>("F0", 1.0));

        string feature;
        while (! iss.eof()) 
        {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string feat = feature.substr(0, findex).c_str();
            if (Tr.feat2idmap.count(feat) == 0) continue;
            float x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
			if (x == 0.0) continue;

            feat2val.push_back(pair<string, float>(feat, x));
        }

        if (feat2val.size() >= 1) 
        {
            Va.Y.push_back(target);

            Va.X.push_back(std::vector<pair<int, float>>());
            int i = Va.X.size() - 1;
            Va.X[i].resize(feat2val.size(), pair<int, float>());
			Va.id2instance[i] = userid;

			string qid = userid//.substr(0, userid.find_first_of("_")).c_str();
			Va.qid2InstMap[qid].push_back(i);


            int j = 0;
        	for (vector<pair<string, float>>::iterator it = feat2val.begin(); it != feat2val.end(); ++it) 
            {
                string feat = it->first;
                float x = it->second;
				int f = Tr.feat2idmap[feat];
                Va.X[i][j] = pair<int, float>(f, x);
                j += 1;
            }
        }
    }
    Va.nr_instance = Va.X.size();
}

bool mysortfunc(const pair<int, float>& a, const pair<int, float>& b)
{
    return a.second > b.second;
}


float calTop1Sim100(Problem& prob, int& predictright, int& predictorder)
{
	for (boost::unordered_map<std::string, vector<int>>::iterator it = prob.qid2InstMap.begin(); it != prob.qid2InstMap.end(); it++)
	{
		string qid = it->first;
		int maxi = -1;
		float maxpred = -100000000;
		for (int t = 0; t < prob.qid2InstMap[qid].size(); t += 1)
		{
			int i = prob.qid2InstMap[qid][t];
            float s = prob.F[i];
			if (s > maxpred)
			{
				maxi = i;
				maxpred = s;
			}
		}
		if (prob.Y[maxi] == 1) predictright += 1;
		predictorder += 1;
	}
	return predictright * 1.0 / predictorder;
}

float calAUC(Problem& prob)
{
    int poscnt = 0;
    vector<pair<int, float>> userid2score(prob.nr_instance);
    map<int, float> userid2label;
    for (int i = 0; i < prob.nr_instance; i += 1)
    {
        float s = prob.F[i];
        userid2score[i] = pair<int, float>(i, s);

        float yi = prob.Y[i];
        if (yi == 1) poscnt += 1;
        userid2label[i] = yi;
    }
    int negcnt = prob.nr_instance - poscnt;
    sort(userid2score.begin(), userid2score.end(), mysortfunc);

    float score = 0;
    int predictposcnt = 0, belownegcnt = negcnt;
    for (size_t i = 0; i < userid2score.size(); ++i) 
    {
        int idx = userid2score[i].first;
        float fscore = userid2score[i].second;
        float yi = userid2label[idx];

        if (yi == 1) 
        {
            score += belownegcnt * 1.0 / negcnt;
            predictposcnt += 1;
        } 
        else {
            belownegcnt -= 1;
        }
    }

    float auc = score * 1.0 / poscnt;
    return auc;
}

int g_seed = 0;
inline int fastrand() { 
	g_seed = (214013*g_seed+2531011); 
	return (g_seed>>16)&0x7FFF; 
} 

float Sqrt (float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(float*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1.0 / x;
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

    cout << "FTRLFMRankV4OMP alpha: " << opt.alpha << "\t nDim: " << opt.ndim << "\tC: " << opt.cls << "\tR: " << opt.rank << "\tP: " << opt.pcnt << endl;

	omp_set_num_threads(20);

	Problem Tr, Va;

    ifstream trainfile(opt.Tr_path);
    loadTrainInstance(Tr, trainfile);
    cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field <<  endl;

    ifstream testfile(opt.Va_path);
    loadTestInstance(Tr, Va, testfile);
    cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;


	float alpha = opt.alpha;
	float beta = opt.beta;
	float lambda_1 = opt.lambda_1;
	float lambda_2 = opt.lambda_2;
	int nDim = opt.ndim;

	Tr.V.resize(Tr.nr_field, vector<float>());
	Tr.VN.resize(Tr.nr_field, vector<float>());
	Tr.VZ.resize(Tr.nr_field, vector<float>());
	#pragma omp parallel for schedule(static)
	for (int i = 0; i < Tr.nr_field; i += 1)
	{
		Tr.V[i].resize(nDim, 0); 
		Tr.VN[i].resize(nDim, 0); 
		Tr.VZ[i].resize(nDim, 0); 

		for (int k = 0; k < nDim; k += 1)
		{
			//float v = (rand() % 100 - 50) /1000.0;
			float v = (rand() % 100000) / 100000.0 - 0.5;
			Tr.V[i][k] = v;
		}
	}


	Tr.W.resize(Tr.nr_field, 0);
	Tr.N.resize(Tr.nr_field, 0);
	Tr.Z.resize(Tr.nr_field, 0);

	Tr.F.resize(Tr.nr_instance, 0);
	Va.F.resize(Va.nr_instance, 0);

    cout << "iter\ttime\trankright\ttrain_top1sim100\ttest_top1sim100\tauc" << endl;
        
	vector<int> inst_idx_vec(Tr.nr_instance);
    for (int i = 0; i < Tr.nr_instance; i += 1) 
		inst_idx_vec[i] = i;
	
	float pcnt = 0.5;
	pcnt = 1.0;
	pcnt = 1.5;
	pcnt = 2;
	pcnt = opt.pcnt;
	for(int iter = 0; iter < opt.nr_iter; iter += 1)
	{
        Timer timer;
		timer.reset();
		timer.tic();

		random_shuffle(inst_idx_vec.begin(),inst_idx_vec.end());

		int instCnt = 0;
		int rankright = 0;
    	for (vector<int>::iterator it = inst_idx_vec.begin(); it < inst_idx_vec.end(); ++it) 
    	{
			int i = *it;
            int yi = Tr.Y[i];

			if (opt.cls == 1)
			{
				float Fi = 0;
				float NF = pow(Tr.X[i].size() - 1, pcnt);
				for (int t = 0; t < Tr.X[i].size(); t += 1) 
                {
					pair<int, float>& inst = Tr.X[i][t];
                	int j = inst.first;
                	float x = inst.second;
					Fi += Tr.W[j] * x;
				}

				float gRes[nDim];
				for (int k = 0; k < nDim; k += 1)
				{
					float tempLeftSum = 0;
					float tempRightSum = 0;
					for (int t = 0; t < Tr.X[i].size(); t += 1)
					{
						pair<int, float>& inst = Tr.X[i][t];
                        int j = inst.first;
						tempLeftSum += Tr.V[j][k];
						tempRightSum += Tr.V[j][k] * Tr.V[j][k];
					}
					Fi += 0.5 * (tempLeftSum * tempLeftSum - tempRightSum) / NF;
					gRes[k] = tempLeftSum;
				}

				Tr.F[i] = Fi;
				float p = 1.0 / (1 + exp(- Tr.F[i]));
				float gLoss = p - yi;

				for (int t = 0; t < Tr.X[i].size(); t += 1)
                {
					pair<int, float>& inst = Tr.X[i][t];
                	int j = inst.first;
                	float x = inst.second;

					float g = gLoss * x;
					float tao = (sqrt(Tr.N[j] + g * g) - sqrt(Tr.N[j])) / alpha;
					Tr.Z[j] = (Tr.Z[j] + g - tao * Tr.W[j]);
					Tr.N[j] = (Tr.N[j] + g * g);
					if (fabs(Tr.Z[j]) <= lambda_1) Tr.W[j] = 0.0; 
					else Tr.W[j] = - (Tr.Z[j] - sign(Tr.Z[j]) * lambda_1) / ((beta + sqrt(Tr.N[j])) / alpha + lambda_2);
				}

				for (int t = 0; t < Tr.X[i].size(); t += 1)
                {
					pair<int, float>& inst = Tr.X[i][t];
                	int j = inst.first;
                	float x = inst.second;
				
					for (int k = 0; k < nDim; k += 1)
					{
                        float g = gLoss * x;
						float g = gLoss * (gRes[k] - Tr.V[j][k]) / NF;
						float tao = (sqrt(Tr.VN[j][k] + g * g) - sqrt(Tr.VN[j][k])) / alpha;
						Tr.VZ[j][k] = (Tr.VZ[j][k] + g - tao * Tr.V[j][k]);
						Tr.VN[j][k] = (Tr.VN[j][k] + g * g);
						Tr.V[j][k] = - (Tr.VZ[j][k] - sign(Tr.VZ[j][k]) * lambda_1) / ((beta + sqrt(Tr.VN[j][k])) / alpha + lambda_2);
					}
				}
			}


			string qid = Tr.id2qid[i];
			if (opt.rank == 0) continue;
			if (yi == 1) continue;
			if (Tr.qid2posInstMap[qid].size() == 0) continue;

			//排序
			int pi = Tr.qid2posInstMap[qid][0];
			float pFi = 0.0;
			float PNF = pow(Tr.X[pi].size() - 1, pcnt);
			map<int, float> posFeatMap;
			for (int t = 0; t < Tr.X[pi].size(); t += 1) 
            {
				pair<int, float>& inst = Tr.X[pi][t];
                int j = inst.first;
                float x = inst.second;
				pFi += Tr.W[j] * x;
				posFeatMap[j] = x;
			}

			map<int, float> posResMap;
			for (int k = 0; k < nDim; k += 1)
			{
				float tempLeftSum = 0;
				float tempRightSum = 0;
				for (int t = 0; t < Tr.X[pi].size(); t += 1) 
				{
					pair<int, float>& inst = Tr.X[pi][t];
                    int j = inst.first;
					tempLeftSum += Tr.V[j][k];
					tempRightSum += Tr.V[j][k] * Tr.V[j][k];
				}
				posResMap[k] = tempLeftSum;
				pFi += 0.5 * (tempLeftSum * tempLeftSum - tempRightSum) / PNF;
			}
			Tr.F[pi] = pFi;

			int ni = i;
			float NNF = pow(Tr.X[ni].size() - 1, pcnt);
			float nFi = 0;

			map<int, float> negFeatMap;
			for (int t = 0; t < Tr.X[ni].size(); t += 1) 
            {
				pair<int, float>& inst = Tr.X[ni][t];
                int j = inst.first;
                float x = inst.second;
				nFi += Tr.W[j] * x;
				negFeatMap[j] = x;
			}

			map<int, float> negResMap;
			for (int k = 0; k < nDim; k += 1)
			{
				float tempLeftSum = 0;
				float tempRightSum = 0;
				for (int t = 0; t < Tr.X[ni].size(); t += 1)
				{
					pair<int, float>& inst = Tr.X[ni][t];
                    int j = inst.first;
					tempLeftSum += Tr.V[j][k];
					tempRightSum += Tr.V[j][k] * Tr.V[j][k];
				}
				negResMap[k] = tempLeftSum;
				nFi += 0.5 * (tempLeftSum * tempLeftSum - tempRightSum) / NNF;
			}
			Tr.F[ni] = nFi;

			float Fi = pFi - nFi;
			float p = 1.0 / (1 + exp(- Fi));
			float gF = p - 1;

			if (Fi > 0) rankright += 1;
			instCnt += 1;

			//if (Fi > 0) continue;

			for (map<int, float>::iterator it = posFeatMap.begin(); it != posFeatMap.end(); ++it)
			{
				int j = it->first;
                float x = it->second;

				float g = 0;
				if (negFeatMap.find(j) == negFeatMap.end()) g = gF * x;
				else g = gF * (posFeatMap[j] - negFeatMap[j]);

				if (fabs(Tr.Z[j]) <= lambda_1) Tr.W[j] = 0.0;
				else Tr.W[j] = - (Tr.Z[j] - sign(Tr.Z[j]) * lambda_1) / ((beta + sqrt(Tr.N[j])) / alpha + lambda_2);
				float tao = (sqrt(Tr.N[j] + g * g) - sqrt(Tr.N[j])) / alpha;
				Tr.Z[j] = (Tr.Z[j] + g - tao * Tr.W[j]);
				Tr.N[j] = (Tr.N[j] + g * g);
			}

			for (map<int, float>::iterator it = negFeatMap.begin(); it != negFeatMap.end(); ++it)
			{
				int j = it->first;
                float x = it->second;
				if (posFeatMap.find(j) != posFeatMap.end()) continue;

				float g = -gF * x;
				if (fabs(Tr.Z[j]) <= lambda_1) Tr.W[j] = 0.0; 
				else Tr.W[j] = - (Tr.Z[j] - sign(Tr.Z[j]) * lambda_1) / ((beta + sqrt(Tr.N[j])) / alpha + lambda_2);
				float tao = (sqrt(Tr.N[j] + g * g) - sqrt(Tr.N[j])) / alpha;
				Tr.Z[j] = (Tr.Z[j] + g - tao * Tr.W[j]);
				Tr.N[j] = (Tr.N[j] + g * g);
			}

			#pragma omp parallel for schedule(static) 
			for (int k = 0; k < nDim; k += 1)
			{
				for (map<int, float>::iterator it = posFeatMap.begin(); it != posFeatMap.end(); ++it)
				{
					int j = it->first;	

					float g = 0;
					if (negFeatMap.find(j) == negFeatMap.end()) g = gF * (posResMap[k] - Tr.V[j][k]) / PNF;
					else g = gF * ((posResMap[k] - Tr.V[j][k]) / PNF - (negResMap[k] - Tr.V[j][k]) / NNF);

					Tr.V[j][k] = - (Tr.VZ[j][k] - sign(Tr.VZ[j][k]) * lambda_1) / ((beta + sqrt(Tr.VN[j][k])) / alpha + lambda_2);
					float tao = (sqrt(Tr.VN[j][k] + g * g) - sqrt(Tr.VN[j][k])) / alpha;
					Tr.VZ[j][k] = (Tr.VZ[j][k] + g - tao * Tr.V[j][k]);
					Tr.VN[j][k] = (Tr.VN[j][k] + g * g);
				}

				for (map<int, float>::iterator it = negFeatMap.begin(); it != negFeatMap.end(); ++it)
				{
					int j = it->first;
					if (posFeatMap.find(j) != posFeatMap.end()) continue;

					float g = -gF * (negResMap[k] - Tr.V[j][k]) / NNF;
					Tr.V[j][k] = - (Tr.VZ[j][k] - sign(Tr.VZ[j][k]) * lambda_1) / ((beta + sqrt(Tr.VN[j][k])) / alpha + lambda_2);
					float tao = (sqrt(Tr.VN[j][k] + g * g) - sqrt(Tr.VN[j][k])) / alpha;
					Tr.VZ[j][k] = (Tr.VZ[j][k] + g - tao * Tr.V[j][k]);
					Tr.VN[j][k] = (Tr.VN[j][k] + g * g);
				}
			}
		}
	
		int trimcnt = 0;
		//#pragma omp parallel for schedule(static) reduction(+: trimcnt)
		for (int j = 0; j < Tr.W.size(); j += 1) 
		{
			if (fabs(Tr.W[j]) <= 0)
				trimcnt += 1;
		}
		float sparity = trimcnt * 100.0 / Tr.nr_field;

    	for (int i = 0; i < Va.nr_instance; i += 1) 
    	{
            float yi = Va.Y[i];
			float Fi = 0;

			float NF = pow(Va.X[i].size() - 1, pcnt);
            for (vector<pair<int, float>>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it)
            {
                int j = it->first;
                float x = it->second;
				Fi += Tr.W[j] * x;
			}

			for (int k = 0; k < nDim; k += 1)
			{
				float tempLeftSum = 0;
				float tempRightSum = 0;
				for (int t = 0; t < Va.X[i].size(); t += 1) 
				{
					pair<int, float>& inst = Va.X[i][t];
                    int j = inst.first;
					tempLeftSum += Tr.V[j][k];
					tempRightSum += Tr.V[j][k] * Tr.V[j][k];
				}
				Fi += 0.5 * (tempLeftSum * tempLeftSum - tempRightSum) / NF;
			}
			Va.F[i] = Fi;
		}

		int train_pr = 0, train_pa = 0;
		float top1sim100_train = calTop1Sim100(Tr, train_pr, train_pa);
		int val_pr = 0, val_pa = 0;
		float top1sim100 = calTop1Sim100(Va, val_pr, val_pa);

		float vaauc = calAUC(Va);
		//float trauc = calAUC(Tr);
		cout << iter << "\t" << timer.toc() << "\t" << rankright * 1.0 / instCnt<< "\t" << top1sim100_train << "\t" << top1sim100 << "(" << val_pr << "," << val_pa << ")\t"<< vaauc << endl; 

		if (iter % 2 == 0 && iter > 0)
			writeWeightFile(Tr, Va, iter);
	}

	writeWeightFile(Tr, Va, opt.nr_iter);
}
