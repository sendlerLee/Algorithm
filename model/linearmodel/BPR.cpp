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
// BPR 分类借的Instance做为样本

using namespace std;

struct Problem
{
	Problem() {}

	int nr_instance, nr_field;

	std::vector<vector<int>> X;

	map<string, int> feat2idmap;
	map<string, int> user2idmap;

	double** pItemLatentFactor;
	double** pUserLatentFactor;

	string Tr_path, Va_out_path;

	int nr_iter, nLevel, nDim;
	double nr_lr, lambda_2;
};

Problem Tr;

string train_help()
{
    return std::string(
"usage: BPR [<options>] <train_path> <validation_output_path>\n"
"\n"
"options:\n"
"-i <nr_iter>: set the number of iteration\n"
"-l <nr_lr>: set the learning rate\n"
"-s <nLevel>: negative sample \n"
"-d <nDim>: set latent factor Dim \n"
"-l2 <lambda_2>: set the reg \n");
}

void parse_option(vector<string> const &args)
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
            Tr.nr_iter = stoi(args[++i]);
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            Tr.nr_lr = stof(args[++i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            Tr.lambda_2 = stof(args[++i]);
        }
	else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            Tr.nLevel = stoi(args[++i]);
        }
	else if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            Tr.nDim = stoi(args[++i]);
        }
        else
            break;
    }

    if(i != argc-2)
        throw std::invalid_argument("invalid command2");

    Tr.Tr_path = args[i++];
    Tr.Va_out_path = args[i++];
}

double bprsgd(Problem& Tr, int userid, int positiveitemid, int negativeitemid)
{
	double posval = 0, negval = 0;
	for (int k = 0; k < Tr.nDim; k += 1) 
	{
		posval += Tr.pUserLatentFactor[userid][k] * Tr.pItemLatentFactor[positiveitemid][k];
		negval += Tr.pUserLatentFactor[userid][k] * Tr.pItemLatentFactor[negativeitemid][k];
	}

	if ((posval - negval) <= 0)
		return posval - negval;

	double weight = 1.0 / (exp(posval - negval) + 1);

	for (int k = 0; k < Tr.nDim; k += 1) 
	{
		double qi = Tr.pItemLatentFactor[positiveitemid][k] - Tr.pItemLatentFactor[negativeitemid][k];
		double qp = Tr.pUserLatentFactor[userid][k];
		double qn = 0 - Tr.pUserLatentFactor[userid][k];

		Tr.pUserLatentFactor[userid][k] += Tr.nr_lr * (weight * qi - Tr.lambda_2 * Tr.pUserLatentFactor[userid][k]);
		Tr.pItemLatentFactor[positiveitemid][k] += Tr.nr_lr * (weight * qp - Tr.lambda_2 * Tr.pItemLatentFactor[positiveitemid][k]);
		Tr.pItemLatentFactor[negativeitemid][k] += Tr.nr_lr * (weight * qn - Tr.lambda_2 * Tr.pItemLatentFactor[negativeitemid][k]);
	}

	return posval - negval;
}

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

int OutputUserItemLatentFactor(Problem& Tr)
{
	ofstream outfile(Tr.Va_out_path);
	for (map<string, int>::iterator it = Tr.user2idmap.begin(); it != Tr.user2idmap.end(); ++it)
	{
		string user = it->first;
		int i = it->second;
		int maxk = 0;
		double maxfactor = -1000000;
		for (int k = 0; k < Tr.nDim; k += 1) 
		{
			if (Tr.pUserLatentFactor[i][k] > maxfactor) 
			{
				maxfactor = Tr.pUserLatentFactor[i][k];
				maxk = k;
			}
		}
		outfile << user << "\t" << maxk << ":" << maxfactor << endl;
	}
	outfile.close();
	return 0;
}

/*
int getUserPredictItem(Problem& Tr)
{
	ofstream outfile(Tr.Va_out_path);
	for (map<string, int>::iterator it = Tr.user2idmap.begin(); it != Tr.user2idmap.end(); ++it)
	{
		string user = it->first;
		int i = it->second;
		outfile << user << " ";

		map<int, double> predictmap;
		for (int j = 0; i < Tr.nr_field; j += 1)
		{
			double val = 0.0;
			for (int k = 0; k < Tr.nDim; k += 1) 
			{
				val += Tr.pUserLatentFactor[i][k] * Tr.pItemLatentFactor[j][k];
			}
		}
			
		outfile << endl;
	}
	outfile.close();
	return 0;
}
*/

void loadTrainInstance(Problem& Tr, ifstream& inputfile)
{
	string line;
	while(getline(inputfile, line)) 
	{
		istringstream iss(line);
		string userid;
		int target;
		iss >> userid >> target;

		int i = Tr.user2idmap.size();
		Tr.user2idmap[userid] = i;
		Tr.X.push_back(vector<int>());

		map<string, int> existfeatmap;
		string feature;
		while (iss) 
		{
			iss >> feature;
			int findex = feature.find_last_of(":");
			string feat = feature.substr(0, findex).c_str();
			double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			if (existfeatmap.count(feat) != 0) continue;
			existfeatmap[feat] = 1;

			int f = -1;
			if (Tr.feat2idmap.count(feat) == 0) 
			{
				int f = Tr.feat2idmap.size();
				Tr.feat2idmap[feat] = f;
			}
			f = Tr.feat2idmap[feat];
			Tr.X[i].push_back(f);
		}
	}
	Tr.nr_instance = Tr.user2idmap.size();
	Tr.nr_field = Tr.feat2idmap.size();
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
		 parse_option(argv_to_args(argc, argv));
	}
	catch(std::invalid_argument const &e) {	
		std::cout << e.what();
		return EXIT_FAILURE;
	}


	ifstream trainfile(Tr.Tr_path);
        loadTrainInstance(Tr, trainfile);

	cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field <<  endl;

	srandom(time(NULL));

	Tr.pUserLatentFactor = new double*[Tr.nr_instance];
	for (int i = 0; i < Tr.nr_instance; i += 1) 
	{
		Tr.pUserLatentFactor[i] = new double[Tr.nDim];
		for (int j = 0; j < Tr.nDim; j += 1) 
		{
			int randval = rand() % 20000 - 10000;
			Tr.pUserLatentFactor[i][j] = randval / 100000.0;
		}
	}

	Tr.pItemLatentFactor = new double*[Tr.nr_field];
	for (int i = 0; i < Tr.nr_field; i += 1) 
	{
		Tr.pItemLatentFactor[i] = new double[Tr.nDim];
		for (int j = 0; j < Tr.nDim; j += 1) 
		{
			int randval = rand() % 20000 - 10000;
			Tr.pItemLatentFactor[i][j] = randval / 100000.0;
		}
	}

	for (int iter = 0; iter < Tr.nr_iter; iter += 1) 
	{
		int rankright = 0;
		int cnt = 0;
		for (int i = 0; i < Tr.X.size(); i += 1) 
		{
			for (int j = 0; j < Tr.X[i].size(); j += 1) 
			{
				int positiveitemid = Tr.X[i][j];
				for (int t = 0; t < Tr.nLevel; t += 1) 
				{
					int k = rand() % Tr.nr_field;
					if (k == positiveitemid) continue;
					double score = bprsgd(Tr, i, j, k);
					if (score > 0) rankright += 1;
					cnt += 1;
				}
            		}
        	}
		cout << iter << "\t" << rankright * 1.0 / cnt << endl;
    	}

	OutputUserItemLatentFactor(Tr);

return 0;
}
