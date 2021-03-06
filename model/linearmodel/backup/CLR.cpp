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
    double ectr;
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
    std::vector<double> W;
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
bool mysortfunc(const pair<string, double>& a, const pair<string, double>& b)
{
    return a.second > b.second;
}

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

double calAUC(Problem& prob)
{
	int poscnt = 0;
	vector<pair<string, double>> userid2score(prob.nr_instance);
	map<string, double> userid2label;
	for (int i = 0; i < prob.nr_instance; i += 1)
	{
		string username = prob.instidmap[i];
		userid2score[i] = pair<string, double>(username, prob.F[i]);

		int yi = prob.Y[i];
		if (yi == 1) poscnt += 1;
		userid2label[username] = yi;
	}
	int negcnt = prob.nr_instance - poscnt;
	sort(userid2score.begin(), userid2score.end(), mysortfunc);

        ofstream outfile(opt.Va_out_path);

	vector<int> pvcnt;
	vector<int> covcnt;
	double score = 0;
	int startpoint = 1000, predictposcnt = 0, belownegcnt = negcnt;
	for (size_t i = 0; i < userid2score.size(); ++i) 
	{
        	string username = userid2score[i].first;
                double fscore = userid2score[i].second;
		int yi = userid2label[username];
                outfile << username << "\t" << fscore << "\t" << yi << endl;

		if (yi == 1) 
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
	pvcnt.push_back(userid2score.size());
	covcnt.push_back(predictposcnt);
	
	//for (int i = 0; i < pvcnt.size(); i += 1) cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

	double auc = score * 1.0 / poscnt;
	return auc;
}
*/

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
		ctr.ectr = ctr.basectr;
	}
		
	smooth = 20;
	for (map<int, CTR>::iterator it = userctrmap.begin(); it != userctrmap.end(); it++) {
		CTR& ctr = it->second;
		ctr.basectr = ctr.clk * 1.0 / (ctr.pv + smooth);
		ctr.ectr = ctr.basectr;
		ctr.ectr = 0;
	}
		
	smooth = 100;
	for (map<int, CTR>::iterator it = adsctrmap.begin(); it != adsctrmap.end(); it++) {
		CTR& ctr = it->second;
		ctr.basectr = ctr.clk * 1.0 / (ctr.pv + smooth);
		ctr.ectr = ctr.basectr;
	}
		
	double trainllh = 0;
        for (int i = 0; i < Tr.nr_instance; i += 1) {
		Instance& ins = Tr.X[i];
		int y = Tr.Y[i];
		double s = userctrmap[ins.u].ectr + pidctrmap[ins.p].ectr * adsctrmap[ins.a].ectr;
		trainllh += y * s - log(1 + exp(s));
	}
	trainllh /= Tr.nr_instance;

	double testllh = 0;
        for (int i = 0; i < Va.nr_instance; i += 1) {
		Instance& ins = Va.X[i];
		int y = Va.Y[i];
		double s = userctrmap[ins.u].ectr + pidctrmap[ins.p].ectr * adsctrmap[ins.a].ectr;
		testllh += y * s - log(1 + exp(s));
	}
	testllh /= Va.nr_instance;
	cout << "Init trainllh: " << trainllh << "\tInit testllh: " << testllh << endl;

	vector<int> trainidxvec(Tr.nr_instance);
        for (int idx = 0; idx < Tr.nr_instance; idx += 1) 
		trainidxvec[idx] = idx;

  	for (int iter = 0; iter < opt.nr_iter; iter += 1)
	{
		std::random_shuffle(trainidxvec.begin(), trainidxvec.end());
        	for (int idx = 0; idx < trainidxvec.size(); idx += 1) 
		{
			int i = trainidxvec[idx];
            		int y = Tr.Y[i];

			int aidx = Tr.X[i].a;
			double a = adsctrmap[aidx].ectr;

			int pidx = Tr.X[i].p;
			double p = pidctrmap[pidx].ectr;

			int uidx = Tr.X[i].u;
			double u = userctrmap[uidx].ectr;

			double g = 0.0;
			double pi = 1.0 / (1 + exp(-(u + p * a)));
			g = y * a - a * pi;
			double nxt_step_p = p + opt.nr_lr * (g + opt.nr_reg * p);

			g = y * p - p * pi;
			double nxt_step_a = a + opt.nr_lr * (g + opt.nr_reg * a);

			g = y  - pi;
			double nxt_step_u = u + opt.nr_lr * (g + opt.nr_reg * u);

			adsctrmap[aidx].ectr = nxt_step_a;
			pidctrmap[pidx].ectr = nxt_step_p;
			//userctrmap[uidx].ectr = nxt_step_u;
		}

		double trainllh = 0;
        	for (int i = 0; i < Tr.nr_instance; i += 1) {
			Instance& ins = Tr.X[i];
			int y = Tr.Y[i];
 
			int p = ins.p;
			int u = ins.u;
			int a = ins.a;
			double s = userctrmap[u].ectr + pidctrmap[p].ectr * adsctrmap[a].ectr;

			trainllh += y * s - log(1 + exp(s));
		}
		trainllh /= Tr.nr_instance;

		double testllh = 0;
        	for (int i = 0; i < Va.nr_instance; i += 1) {
			Instance& ins = Va.X[i];
			int y = Va.Y[i];
			double s = userctrmap[ins.u].ectr + pidctrmap[ins.p].ectr * adsctrmap[ins.a].ectr;
			testllh += y * s - log(1 + exp(s));
		}
		testllh /= Va.nr_instance;
		cout << "Iter: " << iter << "\ttrainllh: " << trainllh << "\tInit testllh: " << testllh << endl << endl;

		//double testauc = calAUC(Va);
		//cout << "Iter: " << iter << "\tTrain llh: " << trainllk << "\tTrainAUC: " << trainauc << "\tTest llh: " << testllh << "\tTestAuc: " << testauc << endl;
	}

	/*
	for (map<int, CTR>::iterator it = pidctrmap.begin(); it != pidctrmap.end(); it++) {
		int p = it->first;
		cout << "pid:\t" << p << "\t" << Tr.id2pidfeatmap[p] << "\t" << pidctrmap[p].ectr << endl;
	}
		
	for (map<int, CTR>::iterator it = userctrmap.begin(); it != userctrmap.end(); it++) {
		int u = it->first;
		cout << "user:\t" << u << "\t" << Tr.id2userfeatmap[u] << "\t" << userctrmap[u].ectr << endl;
	}
		
	for (map<int, CTR>::iterator it = adsctrmap.begin(); it != adsctrmap.end(); it++) {
		int a = it->first;
		cout << "ads:\t" << a << "\t" << Tr.id2adsfeatmap[a] << "\t" << adsctrmap[a].ectr << endl;
	}
	*/
		
return 0;
}
