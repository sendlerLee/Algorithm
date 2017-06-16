#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "lbfgs.h"

using namespace std;


struct Problem
{
    Problem() {}

    int nr_instance, nr_field;
    double sigma;
    map<string, int> feat2idmap;
    map<int, string> id2featmap;

    map<int, string> instidmap;
    vector<map<int, double>> X;

    vector<vector<pair<int, double>>> Xhat;

    vector<double> Y;
    vector<double> F;
    vector<double> B;
    vector<double> W;
};

class objective_function
{
public:
    Problem Tr;
    Problem Va;
    const string va_out_path;
    const string trainfile;
    const string testfile;

    objective_function()
    {
    }

    objective_function(string trainfile,string testfile,string va_out_path) :
	trainfile(trainfile), testfile(testfile), va_out_path(va_out_path) {}

    void loadfile();

    virtual ~objective_function()
    {
    }

    lbfgsfloatval_t getSigma(Problem& Tr) {
	vector<double> resultSet(Tr.nr_instance);
        for (int i = 0; i < Tr.nr_instance; i += 1)
        {
            int yi = Tr.Y[i];
            double bidPrice = Tr.B[i];
            if(yi == 1)	resultSet.push_back(bidPrice);
        }

        double sum = std::accumulate(std::begin(resultSet), std::end(resultSet), 0.0);
        double mean =  sum / resultSet.size(); //均值  

        double accum  = 0.0;
        std::for_each (std::begin(resultSet), std::end(resultSet), [&](const double d) {
                accum  += (d-mean)*(d-mean);
        });


        lbfgsfloatval_t stdev = sqrt(accum/(resultSet.size()-1)); //方差  

        return stdev;

    }
    
    int run(lbfgsfloatval_t orthantwise_c)
    {

	lbfgsfloatval_t fx;
	lbfgsfloatval_t *m_x = lbfgs_malloc(Tr.nr_field);

	lbfgs_parameter_t param;

	for (int f = 0; f < Tr.nr_field; f += 1)
        {
                m_x[f] = 0; 
        }
        /*
            Start the L-BFGS optimization; this will invoke the callback functions
            evaluate() and progress() when necessary.
         */

	lbfgs_parameter_init(&param);
	param.orthantwise_c = orthantwise_c;

        int ret = lbfgs(Tr.nr_field, m_x, &fx, _evaluate, _progress, this, &param);

        /* Report the result. */
        printf("L-BFGS optimization terminated with status code = %d\n", ret);
        
        return ret;
    }

protected:

    static double pnorm(double x)
    {
	double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;

	int sign = 1;
        if (x < 0)
            sign = -1;
    	x = fabs(x)/sqrt(2.0);

	double t = 1.0/(1.0 + p*x);
        double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

        return 0.5*(1.0 + sign*y);
    }

    static double dnorm(double x,double miu, double sigma)
    {
        return 1.0/(sqrt(2*M_PI)*sigma) * exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
    }

    void writeWeightFile(const lbfgsfloatval_t *w) {
	ofstream outfile("UM_PriceModel");
        for (int f = 0; f < Tr.nr_field; f += 1)
        {
            string fname = Tr.id2featmap[f];
	    if(w[f] != 0)
                outfile << "PRICEMODEL\t" << fname << "\t" << w[f] << endl;
        }
        outfile.close();
    }

    void writeResultFile() {
        ofstream outfile(va_out_path);
        for(int f = 0; f < Va.nr_instance; f+= 1) {
                outfile << Va.Y[f] << "\t" << Va.B[f] << "\t" << Va.F[f] << endl;
        }
        outfile.close();
    }

    static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
        return reinterpret_cast<objective_function*>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
        lbfgsfloatval_t fx = 0.0;

	#pragma omp parallel for schedule(static) reduction(+: fx)
        for (int i = 0; i < Tr.nr_instance; i += 1)
        {
	    double Fi = 0.0;
            for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
            {
               int f = it->first;
               double v = it->second;
               Fi += x[f] * v;
            }
	    Tr.F[i] = Fi;
	    double z = (Tr.F[i] - Tr.B[i])/Tr.sigma;
	    //if(i < 5)    cout << "Fi:" << Fi << "\tz:" << z << "\tdnorm:" << log(dnorm(z,0,1)) << "\tpnorm:" << log(pnorm(z)) << endl;
            if(z > 3) z = 3;
            if(z < -3) z = -3;
            fx += - (Tr.Y[i] * log(dnorm(z,0,1)) + (1 - Tr.Y[i]) * log(pnorm(z)));
        }
        
	#pragma omp parallel for schedule(dynamic)
        for (int f = 0; f < Tr.nr_field; f += 1)
        {
	    double gradient = 0.0;
            for (int j = 0; j < Tr.Xhat[f].size(); j += 1)
            {
                pair<int, double>& ins = Tr.Xhat[f][j];
		int i = ins.first;
		double v = ins.second;
		double z = (Tr.F[i] - Tr.B[i])/Tr.sigma;
		if(z > 3) z = 3;
		if(z < -3) z = -3;
		double dpnorm = - exp(log(dnorm(z,0,1)) - log(pnorm(z)));
		gradient += (Tr.Y[i] * z * v / Tr.sigma) +(1 - Tr.Y[i]) * dpnorm * v / Tr.sigma;
            }
	    g[f] = gradient;
        }

	//fx /= Tr.nr_instance;
        //printf("  fx = %f\n", fx);
        return fx;
    }

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        return reinterpret_cast<objective_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
	
	double trainmse = 0.0;
	#pragma omp parallel for schedule(static) reduction(+: trainmse)
	for (int i = 0; i < Tr.nr_instance; i += 1)
	{
		double Fi = 0.0;
		for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
		{
			int f = it->first;
			double v = it->second;
			Fi += x[f] * v;
		}
		Tr.F[i] = Fi;
		double z = (Fi - Tr.B[i])/Tr.sigma;
		if(z < -3) z = -3;
		if(z > 3) z = 3;
		trainmse += (Tr.B[i] - Tr.F[i]) * (Tr.B[i] - Tr.F[i]);
	}
	trainmse /= Tr.nr_instance;

	writeWeightFile(x);

	double testmse = 0;
	double testloss = 0.0;
	#pragma omp parallel for schedule(static) reduction(+: testmse, testloss)
	for (int i = 0; i < Va.nr_instance; i += 1)
	{
		double Fi = 0.0;
		for (map<int, double>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it)
		{
			int f = it->first;
			double v = it->second;
			Fi += x[f] * v;
		}
		Va.F[i] = Fi;
		double z = (Fi - Va.B[i])/Tr.sigma;
		if(z < -3) z = -3;
		if(z > 3) z = 3;
		testloss += - (Va.Y[i] * log(dnorm(z,0,1)) + (1 - Va.Y[i]) * log(pnorm(z)));
		testmse += (Va.B[i] - Va.F[i]) * (Va.B[i] - Va.F[i]);
	}
	testmse /= Va.nr_instance;

	writeResultFile();
	
        printf("Iteration %d:, trainloss = %f, trainmse = %f, testloss = %f, testmse = %f, step = %f\n", k, fx, trainmse, testloss, testmse, step);
        return 0;
    }
};


void objective_function::loadfile() {
    ifstream trFile(trainfile);
    string line;
    while(getline(trFile, line)) {
        istringstream iss(line);
        string userid;
        double bidprice;
        int target;
        iss >> userid >> target >> bidprice;
        if (target != 1) target = 0;

        Tr.Y.push_back(target);
        Tr.B.push_back(bidprice);
        Tr.F.push_back(0);
        Tr.X.push_back(std::map<int, double>());

        int i = Tr.X.size() - 1;
        Tr.instidmap[i] = userid;

        string feature;
        while (iss) {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string feat = feature.substr(0, findex).c_str();
            double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

            int f = -1;
            if (Tr.feat2idmap.count(feat) == 0) {
                // new feature
                int fid = Tr.feat2idmap.size();
                Tr.W.push_back(0);
                Tr.feat2idmap[feat] = fid;
                Tr.id2featmap[fid] = feat;
            }
            f = Tr.feat2idmap[feat];
            if (Tr.X[i].count(f) != 0) continue;
            Tr.X[i][f] = fval;
        }
    }
                                                                                                                                                                         Tr.nr_instance = Tr.X.size();
    Tr.nr_field = Tr.feat2idmap.size();

    Tr.sigma = getSigma(Tr);

    cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << Tr.nr_field << " sigma: "<< Tr.sigma <<  endl;

    Tr.Xhat.resize(Tr.nr_field, vector<pair<int, double>>());
    for (int i = 0; i < Tr.nr_instance; i += 1)
    {
        for (map<int, double>::iterator it = Tr.X[i].begin(); it != Tr.X[i].end(); ++it)
        {
            int f = it->first;
            double x = it->second;
            Tr.Xhat[f].push_back(pair<int, double>(i, x));
        }
    }

    for (int f = 0; f < Tr.nr_field; f += 1) Tr.Xhat[f].shrink_to_fit();


    ifstream vaFile(testfile);
    while(getline(vaFile, line)) {
        istringstream iss(line);
        string userid;
        double bidprice;
        int target;
        iss >> userid >> target >> bidprice;
        if (target != 1) target = 0;

        Va.Y.push_back(target);
        Va.B.push_back(bidprice);
        Va.F.push_back(0);
        Va.X.push_back(std::map<int, double>());

        int i = Va.X.size() - 1;
        Va.instidmap[i] = userid;

        string feature;
        while (iss) {
                iss >> feature;
                int findex = feature.find_last_of(":");
                string feat = feature.substr(0, findex).c_str();
                double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

                if (Tr.feat2idmap.count(feat) == 0) continue;
                int j = Tr.feat2idmap[feat];
                Va.X[i][j] = fval;
        }
    }
    Va.nr_instance = Va.X.size();
    cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;
}

struct Option
{
    Option() : nr_reg(0.01){}
    std::string Tr_path, Va_path, Va_out_path;
    double  nr_reg;
};

Option opt;

std::string train_help()
{
    return std::string(
"usage: Tobit [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-r <nr_reg>: set the reg 0.02\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    int const argc = static_cast<int>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    int i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-r") == 0)
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

std::vector<std::string> argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}

int main(int argc,char const * const * const argv)
{

    try {
	opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e) {
	std::cout << e.what();
	return EXIT_FAILURE;
    }


    omp_set_num_threads(static_cast<int>(50));

    objective_function *obj = new objective_function(opt.Tr_path,opt.Va_path,opt.Va_out_path);
    obj->loadfile();
    obj->run(opt.nr_reg);
}

