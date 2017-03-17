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

struct CTR
{
    CTR() {}
    int pv; 
    int clk;
    double basectr;
    double x;
};

struct Instance
{
    Instance() {}

    int u;
    int p;
    int a;
};

struct Problem
{
    Problem() {}

    int nr_instance, nr_field;

    std::map<std::string, int> userfeat2idmap;
    std::map<int, std::string> id2userfeatmap;

    std::map<std::string, int> pidfeat2idmap;
    std::map<int, std::string> id2pidfeatmap;

    std::map<std::string, int> adsfeat2idmap;
    std::map<int, std::string> id2adsfeatmap;

    std::vector<Instance> X;

    std::vector<double> Y;
    std::vector<double> F;
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

/*
void writeWeightFile(Problem& Tr)
{
	ofstream outfile("feature_weight_lr");
	for (int f = 0; f < Tr.nr_field; f += 1)
	{
		string fname = Tr.id2featmap[f];
		outfile << fname << " " << Tr.W[f] << endl;
        }
	outfile.close();
}
*/

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

double calAUC(Problem& prob)
{
	int poscnt = 0;
	vector<pair<int, double>> userid2score(prob.nr_instance);
	map<int, double> userid2label;
	for (int i = 0; i < prob.nr_instance; i += 1)
	{
		userid2score[i] = pair<int, double>(i, prob.F[i]);

		int yi = prob.Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[i] = yi;
	}
	int negcnt = prob.nr_instance - poscnt;
	sort(userid2score.begin(), userid2score.end(), mysortfunc);

        ofstream outfile(opt.Va_out_path);

	double avgidx = 0;
	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < userid2score.size(); ++i) 
	{
        	int idx = userid2score[i].first;
                double fscore = userid2score[i].second;
		int yi = userid2label[idx];
                outfile << idx << "\t" << fscore << "\t" << yi << endl;

		if (yi == 1) 
		{
			avgidx += i;
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
	cout << "Average Index: " << avgidx * 1.0 / userid2score.size() << "\t" << userid2score.size() << endl;
	
	for (int i = 0; i < pvcnt.size(); i += 1) cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	return auc;
}

void loadTrainInstance(Problem& Tr, ifstream& inputfile) 
{    
	string line;
	while(getline(inputfile, line)) {
		istringstream iss(line);
		string userid, pid, ads;
		int target;
		iss >> target >> userid >> pid >> ads;
		if (target != 1) target = 0;

		Tr.Y.push_back(target);

		Instance ins;

		if (Tr.userfeat2idmap.count(userid) == 0) {
			int fid = Tr.userfeat2idmap.size();
			Tr.userfeat2idmap[userid] = fid;
			Tr.id2userfeatmap[fid] = userid;
		}
		ins.u = Tr.userfeat2idmap[userid];

		if (Tr.pidfeat2idmap.count(pid) == 0) {
			int fid = Tr.pidfeat2idmap.size();
			Tr.pidfeat2idmap[pid] = fid;
			Tr.id2pidfeatmap[fid] = pid;
		}
		ins.p = Tr.pidfeat2idmap[pid];

		if (Tr.adsfeat2idmap.count(ads) == 0) {
			int fid = Tr.adsfeat2idmap.size();
			Tr.adsfeat2idmap[ads] = fid;
			Tr.id2adsfeatmap[fid] = ads;
		}
		ins.a = Tr.adsfeat2idmap[ads];
		Tr.X.push_back(ins);
		Tr.F.push_back(0);
	}
    		
	Tr.nr_instance = Tr.X.size();
}

void loadTestInstance(Problem& Tr, Problem& Va, ifstream& inputfile) 
{
	string line;
	while(getline(inputfile, line)) {
		istringstream iss(line);
		string userid, pid, ads;
		int target;
		iss >> target >> userid >> pid >> ads;
		if (target != 1) target = 0;

		Va.Y.push_back(target);

		Instance ins;

		ins.u = -1;
		if (Tr.userfeat2idmap.count(userid) != 0) {
			int fid = Tr.userfeat2idmap[userid];
			ins.u = fid;
		}

		ins.p = -1;
		if (Tr.pidfeat2idmap.count(pid) != 0) {
			int fid = Tr.pidfeat2idmap[pid];
			ins.p = fid;
		}

		ins.a = -1;
		if (Tr.adsfeat2idmap.count(ads) != 0) {
			int fid = Tr.adsfeat2idmap[ads];
			ins.a = fid;
		}
		Va.X.push_back(ins);
		Va.F.push_back(0);
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
	cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " User: " << Tr.userfeat2idmap.size() << " PID: " << Tr.pidfeat2idmap.size() << " Ads: " << Tr.adsfeat2idmap.size() << endl;

	ifstream testfile(opt.Va_path);
	loadTestInstance(Tr, Va, testfile);
	cout << "Load Data Finish, Instance:" << Va.nr_instance << " User: " << Va.userfeat2idmap.size() << " PID: " << Va.pidfeat2idmap.size() << " Ads: " << Va.adsfeat2idmap.size() << endl;
	
	map<int, CTR> pidctrmap;
	map<int, CTR> userctrmap;
	map<int, CTR> adsctrmap;
        for (int i = 0; i < Tr.nr_instance; i += 1) {
		int y = Tr.Y[i];
		Instance& ins = Tr.X[i];

		pidctrmap[ins.p].pv += 1;
		pidctrmap[ins.p].clk += y;
		
		userctrmap[ins.u].pv += 1;
		userctrmap[ins.u].clk += y;
		
		adsctrmap[ins.a].pv += 1;
		adsctrmap[ins.a].clk += y;
	}

	int smooth = 100;
	for (map<int, CTR>::iterator it = pidctrmap.begin(); it != pidctrmap.end(); it++) {
		CTR& ctr = it->second;
		ctr.basectr = ctr.clk * 1.0 / (ctr.pv + smooth);
		ctr.x = ctr.basectr;
		ctr.x = 0;
	}
		
	smooth = 20;
	for (map<int, CTR>::iterator it = userctrmap.begin(); it != userctrmap.end(); it++) {
		CTR& ctr = it->second;
		ctr.basectr = ctr.clk * 1.0 / (ctr.pv + smooth);
		ctr.x = 0;
	}
		
	smooth = 100;
	for (map<int, CTR>::iterator it = adsctrmap.begin(); it != adsctrmap.end(); it++) {
		CTR& ctr = it->second;
		ctr.basectr = ctr.clk * 1.0 / (ctr.pv + smooth);
		ctr.x = 0;
	}
		
	double trainrmse = 0;
        for (int i = 0; i < Tr.nr_instance; i += 1) {
		Instance& ins = Tr.X[i];
		int y = Tr.Y[i];
		double s = pidctrmap[ins.p].basectr;
		//trainrmse += (y - s) * (y - s);
		trainrmse += abs(y - s);
		Tr.F[i] = s;
	}
	trainrmse /= Tr.nr_instance;
	
	//double train_auc = calAUC(Tr);
	//cout << "Train_AUC: " << train_auc << endl;

	map<int, double> adseclickmap;
	map<int, int> adstclickmap;
	map<int, int> adsmap;

	double tclick = 0, eclick = 0;
	double testrmse = 0;
        for (int i = 0; i < Va.nr_instance; i += 1) {
		Instance& ins = Va.X[i];
		int y = Va.Y[i];
		double s = pidctrmap[ins.p].basectr;
		//testrmse += (y - s) * (y - s);
		testrmse += abs(y - s);
		Va.F[i] = s;

		eclick += s;
		tclick += y;

		int a = ins.a;
		adseclickmap[a] += s;
		adstclickmap[a] += y;
		adsmap[a] = 1;
	}
	testrmse /= Va.nr_instance;
	double auc = calAUC(Va);
	cout << "Init trainrmse: " << trainrmse << "\tInit testrmse: " << testrmse << "\tAUC: " << auc << "\teclick: " << eclick << "\ttclick: " << tclick << "\trate: " << tclick * 1.0 / eclick << endl;

	int sumdiff = 0;
	for (map<int, int>::iterator it = adsmap.begin(); it != adsmap.end(); it++) {
		int a= it->first;
		sumdiff += abs(adstclickmap[a] - adseclickmap[a]);
	}
	cout << "SumDiff:\t" << sumdiff << endl;
	

	vector<int> trainidxvec(Tr.nr_instance);
        for (int idx = 0; idx < Tr.nr_instance; idx += 1) 
		trainidxvec[idx] = idx;

  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		//std::random_shuffle(trainidxvec.begin(), trainidxvec.end());
        	//for (int idx = 0; idx < trainidxvec.size(); idx += 1) 
        	for (int i = 0; i < Tr.nr_instance; i += 1)
		{
			//int i = trainidxvec[idx];
            		int y = Tr.Y[i];

			Instance& ins = Tr.X[i];
			double a = adsctrmap[ins.a].x;
			double p = pidctrmap[ins.p].x;
			double u = userctrmap[ins.u].x;

			double g = 0.0;

			//double s = a + p + p * a + u;
			double s = a + p + u + pidctrmap[ins.p].x;
			if (y - s > 0) g = -1;
			else g = 1;
			
			//g = - (y - s) * (1 + a) + opt.nr_reg * p;
			g = g + opt.nr_reg * p;
			double nxt_step_p = p - opt.nr_lr * g;

			//g = - (y - s) * (1 + p) + opt.nr_reg * a;
			g = g + opt.nr_reg * a;
			double nxt_step_a = a - opt.nr_lr * g;

			//g = - (y - s) + opt.nr_reg * u;
			g = g + opt.nr_reg * u;
			double nxt_step_u = u - opt.nr_lr * g;

			/*
			adsctrmap[ins.a].x = nxt_step_a;
			pidctrmap[ins.p].x = nxt_step_p;
			userctrmap[ins.u].x = nxt_step_u;
			*/

			if (nxt_step_a > 0 && nxt_step_a < 1) adsctrmap[ins.a].x = nxt_step_a;
			if (nxt_step_p > 0 && nxt_step_p < 1) pidctrmap[ins.p].x = nxt_step_p;
			if (nxt_step_u > 0 && nxt_step_u < 1) userctrmap[ins.u].x = nxt_step_u;

			/*
			if (adsctrmap[ins.a].x < 0) adsctrmap[ins.a].x = 0;
			if (adsctrmap[ins.a].x > 1) adsctrmap[ins.a].x = 1;

			if (pidctrmap[ins.p].x < 0) pidctrmap[ins.p].x = 0;
			if (pidctrmap[ins.p].x > 1) pidctrmap[ins.p].x = 1;

			if (userctrmap[ins.u].x < 0) userctrmap[ins.u].x = 0;
			if (userctrmap[ins.u].x > 1) userctrmap[ins.u].x = 1;
			*/
		}

		double trainrmse = 0;
        	for (int i = 0; i < Tr.nr_instance; i += 1) {
			Instance& ins = Tr.X[i];
			int y = Tr.Y[i];
			//double s = pidctrmap[ins.p].x + userctrmap[ins.u].x + adsctrmap[ins.a].x + pidctrmap[ins.p].x * adsctrmap[ins.a].x;
			double s = pidctrmap[ins.p].x + userctrmap[ins.u].x + adsctrmap[ins.a].x;
			//if (s > 1) s = 1;
			//if (s < 0) s = 0;
			//trainrmse += (y - s) * (y - s);
			trainrmse += abs(y - s);

			double p = s;
			Tr.F[i] = p;
		}
		trainrmse /= Tr.nr_instance;
		
		//double train_auc = calAUC(Tr);
		//cout << "Iter: " << iter << "\tTrain_AUC: " << train_auc << endl;

		map<int, double> adseclickmap;
		map<int, int> adstclickmap;
		map<int, int> adsmap;

		double tclick = 0, eclick = 0;
		double testrmse = 0;
        	for (int i = 0; i < Va.nr_instance; i += 1) {
			Instance& ins = Va.X[i];
			int y = Va.Y[i];
			//double s = pidctrmap[ins.p].x + userctrmap[ins.u].x + adsctrmap[ins.a].x + pidctrmap[ins.p].x * adsctrmap[ins.a].x;
			double s = pidctrmap[ins.p].x + userctrmap[ins.u].x + adsctrmap[ins.a].x;
			//if (s > 1) s = 1;
			//if (s < 0) s = 0;
			//testrmse += (y - s) * (y - s);
			testrmse += abs(y - s);

			double p = s;
			Va.F[i] = p;

			eclick += p;
			tclick += y;

			int a = ins.a;
			adseclickmap[a] += s;
			adstclickmap[a] += y;
			adsmap[a] = 1;
		}
		testrmse /= Va.nr_instance;
		double auc = calAUC(Va);
		cout << "Iter: " << iter << "\ttrain rmse: " << trainrmse << "\ttest rmse: " << testrmse << "\tAUC: " << auc << "\teclick: " << eclick << "\ttclick: " << tclick << "\trate: " << tclick * 1.0 / eclick << endl << endl;

		int sumdiff = 0;
		for (map<int, int>::iterator it = adsmap.begin(); it != adsmap.end(); it++) {
			int a= it->first;
			sumdiff += abs(adstclickmap[a] - adseclickmap[a]);
		}
		cout << "SumDiff:\t" << sumdiff << endl;
	


		//double testauc = calAUC(Va);
		//cout << "Iter: " << iter << "\tTrain llh: " << trainllk << "\tTrainAUC: " << trainauc << "\tTest llh: " << testrmse << "\tTestAuc: " << testauc << endl;
	}


	for (map<int, CTR>::iterator it = pidctrmap.begin(); it != pidctrmap.end(); it++) {
		int p = it->first;
		cout << "pid:\t" << p << "\t" << Tr.id2pidfeatmap[p] << "\t" << pidctrmap[p].x << endl;
	}
		
	for (map<int, CTR>::iterator it = userctrmap.begin(); it != userctrmap.end(); it++) {
		int u = it->first;
		cout << "user:\t" << u << "\t" << Tr.id2userfeatmap[u] << "\t" << userctrmap[u].x << endl;
	}
		
	for (map<int, CTR>::iterator it = adsctrmap.begin(); it != adsctrmap.end(); it++) {
		int a = it->first;
		cout << "ads:\t" << a << "\t" << Tr.id2adsfeatmap[a] << "\t" << adsctrmap[a].x << endl;
	}
		
return 0;
}
