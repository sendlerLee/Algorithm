#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>
#include <stdlib.h>

#include "gbdt.h"
#include "timer.h"

using namespace std;

void lrinit(Problem& Tr, Problem& Va, vector<double>& F_Tr, vector<double>& F_Va)
{
	map<uint32_t, double> featweightmap;
	for(map<uint32_t, string>::iterator it = Tr.id2featmap.begin(); it != Tr.id2featmap.end(); ++it) 
	{
		uint32_t fid = it->first;
		featweightmap[fid] = 0;
	}

	double nr_lr = 0.0001;
	double nr_reg = 0.0001;
	uint32_t nr_iter = 15;

	for (uint32_t iter = 0; iter < nr_iter; iter += 1) 
	{
		for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
		{
			double yi = (Tr.Y[i] + 1) / 2;

			double wxi = 0.0;
			map<uint32_t, double>& featuredict = Tr.inst2features[i];
			for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				uint32_t fid = fit->first;
				double fval = fit->second;
				wxi += featweightmap[fid] * fval;
			}
			double pi = 1 / (1 + exp(- wxi));
			double error = yi - pi;

			for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				uint32_t fid = fit->first;
				double fval = fit->second;
				featweightmap[fid] += nr_lr * (error * fval + nr_reg * featweightmap[fid]);
			}
		}

		double llh = 0.0;
		for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
		{
			double yi = (Tr.Y[i] + 1) / 2;

			double wxi = 0.0;
			map<uint32_t, double>& featuredict = Tr.inst2features[i];
			for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				uint32_t fid = fit->first;
				double fval = fit->second;
				wxi += featweightmap[fid] * fval;
			}
			double pi = 1 / (1 + exp(- wxi));

			llh += yi * log(pi) + (1 - yi) * log(1 - pi);
		}
		llh /= Tr.nr_instance;
		cout << "Iter: " << iter << "\tlikelihood:\t" << llh << endl;
	}

	for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
	{
		double wxi = 0.0;
		map<uint32_t, double>& featuredict = Tr.inst2features[i];
		for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
		{
			uint32_t fid = fit->first;
			double fval = fit->second;
			wxi += featweightmap[fid] * fval;
		}
		double pi = 1 / (1 + exp(- wxi));
		double val = 2 * pi - 1;
		F_Tr[i] = val;
	}

	for(uint32_t i = 0; i < Va.nr_instance; ++i) 
	{
		double wxi = 0.0;
		map<uint32_t, double>& featuredict = Va.inst2features[i];
		for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
		{
			uint32_t fid = fit->first;
			double fval = fit->second;
			wxi += featweightmap[fid] * fval;
		}
		double pi = 1 / (1 + exp(- wxi));
		double val = 2 * pi - 1;
		F_Va[i] = val;
	}

}

double calc_bias(std::vector<double> const &Y)
{
    double y_bar = std::accumulate(Y.begin(), Y.end(), 0.0);
    y_bar /= static_cast<double>(Y.size());
    return static_cast<double>(0.5 * log((1.0+y_bar)/(1.0-y_bar)));
}
 
struct Location
{
    Location() : tnode_idx(1), r(0) {}
    uint32_t tnode_idx;
    double r;
    bool ofb;
};

struct Meta
{
    Meta() : sl(0), s(0), nl(0), n(0), v(-10000000) {}
    double sl, s;
    uint32_t nl, n;
    double v;
};

struct Defender
{
    Defender() : ese(0), threshold(0) {}
    double ese;
    double threshold;
};

void scan(
    Problem const &prob,
    std::vector<Location> const &instlocations,
    std::vector<TreeNode> &tnodes,
    std::vector<Meta> const &metas0,
    std::vector<std::vector<Defender>> &defenders,
    uint32_t const offset, map<uint32_t, uint32_t>& selectFeatMap)
{
    uint32_t minNodeCapacity = prob.minNodeCount;
    #pragma omp parallel for schedule(dynamic)
    for(uint32_t fid = 0; fid < prob.nr_field; fid += 1)
    {
	const std::vector<std::pair<uint32_t, double>>& instancevec = prob.feat2sortinstances[fid];
        std::vector<Meta> metas = metas0;

	if (selectFeatMap[fid] == 0) continue;

	// Instance sort descending order
	for(uint32_t i = 0; i < instancevec.size(); i += 1)
	{
            const std::pair<uint32_t, double> & inst = instancevec[i];
	    uint32_t instid = inst.first;
	    double instval = inst.second;

            Location const &location = instlocations[instid];
	    if(location.ofb == true) continue;

            uint32_t tnode_idx = location.tnode_idx;
            if(tnodes[tnode_idx].shrinked) continue;

            uint32_t f = location.tnode_idx-offset;
            Meta &meta = metas[f];

            if(instval != meta.v && meta.v != -10000000)
            {
                double const sr = meta.s - meta.sl;
                uint32_t const nr = meta.n - meta.nl;
		if (nr > minNodeCapacity && meta.nl > minNodeCapacity)
		{
                	double const current_ese = (meta.sl*meta.sl)/static_cast<double>(meta.nl) + (sr*sr)/static_cast<double>(nr);
                	Defender &defender = defenders[f][fid];
                	double &best_ese = defender.ese;
                	if(current_ese > best_ese)
                	{
                    		best_ese = current_ese;
                    		defender.threshold = meta.v;
                	}
		}
            }
	
            meta.sl += location.r;
            ++meta.nl;
	    meta.v = instval;
	
            if(i == instancevec.size() - 1)
            {
                double const sr = meta.s - meta.sl;
                uint32_t const nr = meta.n - meta.nl;
		if (nr > minNodeCapacity && meta.nl > minNodeCapacity)
		{
                	double const current_ese = (meta.sl*meta.sl)/static_cast<double>(meta.nl) + (sr*sr)/static_cast<double>(nr);
                	Defender &defender = defenders[f][fid];
                	double &best_ese = defender.ese;
                	if(current_ese > best_ese)
                	{
                    		best_ese = current_ese;
                    		defender.threshold = meta.v;
                	}
		}
            }
	}
   }
}

std::mutex CART::mtx;
bool CART::verbose = false;

void CART::fit(Problem &prob, std::vector<double> &R, std::vector<double> &F1)
{
    uint32_t const nr_field = prob.nr_field;
    uint32_t const nr_instance = prob.nr_instance;

    srandom(time(NULL));
    std::vector<Location> locations(nr_instance);
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
    {
        locations[i].r = R[i];
	uint32_t randval = rand();
        bool ofb = static_cast<bool>(rand() % 2);
	locations[i].ofb = ofb;
    }
	
    srandom(time(NULL));
    map<uint32_t, uint32_t> selectFeatMap;
    for(uint32_t fid = 0; fid < prob.nr_field; fid += 1)
    {
	uint32_t ofb = static_cast<uint32_t>(rand() % prob.nr_field);
	if (ofb * 1.0 < 0.5 * prob.nr_field) selectFeatMap[fid] = 0;
	else selectFeatMap[fid] = 1;
    }

    for(uint32_t d = 0, offset = 1; d < max_depth; ++d, offset *= 2)
    {
        uint32_t const nr_leaf = static_cast<uint32_t>(pow(2, d));
        std::vector<Meta> metas0(nr_leaf);

        for(uint32_t i = 0; i < nr_instance; ++i)
        {
            Location &location = locations[i];
	    if(location.ofb == true) continue;
            if(tnodes[location.tnode_idx].shrinked) continue;

            Meta &meta = metas0[location.tnode_idx-offset];
            meta.s += location.r;
            ++meta.n;
        }

        std::vector<std::vector<Defender>> defenders(nr_leaf);
        for(uint32_t f = 0; f < nr_leaf; ++f)
	    defenders[f].resize(nr_field);

        for(uint32_t f = 0; f < nr_leaf; ++f)
        {
            Meta const &meta = metas0[f];
            double const ese = meta.s*meta.s/static_cast<double>(meta.n);
            for(uint32_t j = 0; j < nr_field; ++j)
                defenders[f][j].ese = ese;
        }

	scan(prob, locations, tnodes, metas0, defenders, offset, selectFeatMap);

        for(uint32_t f = 0; f < nr_leaf; ++f)
        {
            Meta const &meta = metas0[f];
            double best_ese = meta.s*meta.s/static_cast<double>(meta.n);
            TreeNode &tnode = tnodes[f+offset];
            for(uint32_t j = 0; j < nr_field; ++j)
            {
                Defender defender = defenders[f][j];
                if(defender.ese > best_ese)
                {
                    best_ese = defender.ese;
                    tnode.feature = j;
                    tnode.threshold = defender.threshold;
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < nr_instance; ++i)
        {
            Location &location = locations[i];

            uint32_t tnode_idx = location.tnode_idx;
            TreeNode &tnode = tnodes[tnode_idx];

            if(tnode.feature == -1) tnode.shrinked = true;
            if(tnode.shrinked) continue;

	    if(prob.inst2features[i].count(tnode.feature) == 0)
		tnode_idx = 2*tnode_idx;
	    else if(prob.inst2features[i][tnode.feature] < tnode.threshold)
		tnode_idx = 2*tnode_idx;
	    else
		tnode_idx = 2*tnode_idx+1; 

	    location.tnode_idx = tnode_idx;
        }
    }

    std::vector<std::pair<double, double>> tmp(max_tnodes, std::make_pair(0, 0));
    for(uint32_t i = 0; i < nr_instance; ++i)
    {
        Location &location = locations[i];
	if(location.ofb == true) continue;

        double const r = locations[i].r;
        uint32_t const tnode_idx = locations[i].tnode_idx;
	tnodes[tnode_idx].capacity += 1;
        tmp[tnode_idx].first += r;
        tmp[tnode_idx].second += fabs(r)*(2-fabs(r));
    }

    std::map<uint32_t, std::string> curfid2feat = prob.id2featmap;
    for(uint32_t tnode_idx = 1; tnode_idx < max_tnodes; ++tnode_idx)
    {
        double a, b;
        std::tie(a, b) = tmp[tnode_idx];
        tnodes[tnode_idx].gamma = (b <= 1e-12)? 0 : static_cast<double>(a/b);
    }

    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
        F1[i] = tnodes[locations[i].tnode_idx].gamma;
}

std::pair<uint32_t, double> CART::predict(std::map<uint32_t, double>& x) 
{
    uint32_t tnode_idx = 1;
    for(uint32_t d = 0; d <= max_depth; ++d)
    {
        TreeNode const &tnode = tnodes[tnode_idx];
        if(tnode.feature == -1)
            return std::make_pair(tnode.idx, tnode.gamma);

        if(x.count(tnode.feature) == 0)
            tnode_idx = tnode_idx*2;
        else if(x[tnode.feature] < tnode.threshold)
            tnode_idx = tnode_idx*2;
        else
            tnode_idx = tnode_idx*2+1;
    }

    return std::make_pair(-1, -1);
}

void GBDT::write(Problem &prob, vector<double> & F_Val, std::string const &path)
{
    uint32_t poscnt = 0;
    vector<std::pair<uint32_t, double>> instscorevec(prob.nr_instance);
    #pragma omp parallel for schedule(static) reduction(+: poscnt)
    for(uint32_t i = 0; i < prob.nr_instance; ++i)
    {
	if(prob.Y[i] == 1) poscnt += 1;
        instscorevec[i] = pair<uint32_t, double>(i, F_Val[i]);
    }
    std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v());
    uint32_t negcnt = prob.nr_instance - poscnt;

    vector<uint32_t> pvcnt;
    vector<uint32_t> covcnt;
    double score = 0;
    double avgidx = 0;
    uint32_t startpoint = 1000;
    uint32_t predictposcnt = 0;
    uint32_t belownegcnt = negcnt;
    std::ofstream outfile(path);
    for(uint32_t i = 0; i < instscorevec.size(); ++i)
    {
	uint32_t instid = instscorevec[i].first;
	string instance = prob.instidmap[instid];

	double val = instscorevec[i].second;
	double label = prob.Y[instid];
	outfile << instance << "\t" << label << "\t1:1\t2:" << val << std::endl;

	if (label == 1) {
                score += belownegcnt * 1.0 / negcnt;
		avgidx += i * 1.0 / poscnt;
		predictposcnt += 1;
        } else {
                belownegcnt -= 1;
        }

	if (i == startpoint) {
		pvcnt.push_back(i);
		covcnt.push_back(predictposcnt);
		startpoint *= 2;
		//startpoint += 1000;
	}
     }
     pvcnt.push_back(instscorevec.size());
     covcnt.push_back(predictposcnt);
     outfile.close();

     cout << "avgidx: " << avgidx * 1.0 / instscorevec.size() << " " << instscorevec.size() << " IDX: " << avgidx * 1.0 / (instscorevec.size()) << endl;
     for (uint32_t i = 0; i < pvcnt.size(); i += 1)
	cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

     double auc = score * 1.0 / poscnt;
     cout << "AUC: " << auc << endl << endl;
}

void GBDT::fit(Problem &Tr, Problem &Va)
{
    bias = calc_bias(Tr.Y);
    std::vector<double> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    //std::vector<double> F_Tr(Tr.nr_instance, 0), F_Va(Va.nr_instance, 0);
    //lrinit(Tr, Va, F_Tr, F_Va);

    Timer timer;
    printf("iter     time    tr_loss    va_loss    tr_mse    va_mse\n");
    for(uint32_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<double> const &Y = Tr.Y;
        std::vector<double> R(Tr.nr_instance), F1(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
            R[i] = static_cast<double>(2*Y[i]/(1+exp(2*Y[i]*F_Tr[i])));

        uint32_t treedepth = 1 + (t / 10);
        if(treedepth > this->max_depth) treedepth = this->max_depth;
        //trees[t].init(treedepth);

        trees[t].init(this->max_depth);
        trees[t].fit(Tr, R, F1);

        double tr_logloss = 0;
        double Tr_mse = 0;
        #pragma omp parallel for schedule(static) reduction(+: tr_logloss, Tr_mse)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += learningrate * F1[i];
	    tr_logloss += log(1 + exp(- Tr.Y[i] * F_Tr[i]));

	    double yi = (Tr.Y[i] + 1) / 2;
	    double pi = 1.0 / (1 + exp(- F_Tr[i]));
	    Tr_mse += (yi - pi) * (yi - pi);
        }
        tr_logloss /= static_cast<double>(Tr.nr_instance);
        Tr_mse /= static_cast<double>(Tr.nr_instance);

        double va_logloss = 0;
        double Va_mse = 0;
        #pragma omp parallel for schedule(static) reduction(+: va_logloss, Va_mse)
        for(uint32_t i = 0; i < Va.nr_instance; ++i)
        {
            std::map<uint32_t, double> x = Va.inst2features[i];
            std::pair<uint32_t, double> res = trees[t].predict(x);
            F_Va[i] += learningrate * res.second;
	    va_logloss += log(1 + exp(- Va.Y[i] * F_Va[i]));
	
	    double yi = (Va.Y[i] + 1) / 2;
	    double pi = 1.0 / (1 + exp(- F_Va[i]));
	    Va_mse += (yi - pi) * (yi - pi);
        }
        va_logloss /= static_cast<double>(Va.nr_instance);
        Va_mse /= static_cast<double>(Va.nr_instance);

	std::cout << t << "\t" << timer.toc() / (t + 1) << "\t" << tr_logloss << "\t" << va_logloss << "\t" << Tr_mse << "\t" << Va_mse << endl;
    }
    write(Va, F_Va, Va_out_path);
    if (gbdtfeat == 1) 
	getGBDTFeat(Tr, Va);
}

double GBDT::getGBDTFeat(Problem &Tr, Problem &Va) 
{
    std::ofstream trainfeatfile(Tr.input_path + ".gbdtfeat");
    for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
    {
	std::map<uint32_t, double> x = Tr.inst2features[i];
	std::vector<uint32_t> idxvec = get_indices(x);
	string inst = Tr.instidmap[i];
	trainfeatfile << inst << "\t" << Tr.Y[i] << "\t";

	map<uint32_t, double>& featuredict = Tr.inst2features[i];
	for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
	{
		uint32_t fid = fit->first;
		string feat = Tr.id2featmap[fid];
		double fval = fit->second;
		trainfeatfile << feat << ":" << fval << "\t";
	}

	for(int j = 0; j < idxvec.size(); j += 1)
		trainfeatfile << "tr" << j << "no" << idxvec[j] << ":1\t";

	trainfeatfile << endl;
    }

    std::ofstream testfeatfile(Va.input_path + ".gbdtfeat");
    for(uint32_t i = 0; i < Va.nr_instance; ++i) 
    {
	std::map<uint32_t, double> x = Va.inst2features[i];
	std::vector<uint32_t> idxvec = get_indices(x);
	string inst = Va.instidmap[i];
	testfeatfile << inst << "\t" << Va.Y[i] << "\t";

	map<uint32_t, double>& featuredict = Va.inst2features[i];
	for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
	{
		uint32_t fid = fit->first;
		string feat = Tr.id2featmap[fid];
		double fval = fit->second;
		testfeatfile << feat << ":" << fval << "\t";
	}

	for(int j = 0; j < idxvec.size(); j += 1)
		testfeatfile << "tr" << j << "no" << idxvec[j] << ":1\t";
	testfeatfile << endl;
    }

    return 0;
}

double GBDT::predict(std::map<uint32_t, double>& x) 
{
    double s = bias;
    for(uint32_t i = 0; i < trees.size(); i += 1)
    {
        std::pair<uint32_t, double> res = trees[i].predict(x);
        s += learningrate * res.second;
    }
    return s;
}

std::vector<uint32_t> GBDT::get_indices(std::map<uint32_t, double>& x) 
{
    uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

    std::vector<uint32_t> indices(nr_tree);
    for(uint32_t t = 0; t < nr_tree; ++t)
        indices[t] = trees[t].predict(x).first;
    return indices;
}
