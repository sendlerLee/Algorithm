#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>

#include "gbdt.h"
#include "timer.h"

using namespace std;

double calc_bias(std::vector<double> const &Y)
{
    double y_bar = std::accumulate(Y.begin(), Y.end(), 0.0);
    y_bar /= static_cast<double>(Y.size());
    //return static_cast<double>(log((1.0+y_bar)/(1.0-y_bar)));
    return static_cast<double>(0.5 * log((1.0+y_bar)/(1.0-y_bar)));
}

struct Location
{
    Location() : tnode_idx(1), r(0) {}
    uint32_t tnode_idx;
    double r;
};

struct Meta
{
    //Meta() : sl(0), s(0), nl(0), n(0), v(0.0f/0.0f) {}
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
    uint32_t const offset, uint32_t thread_idx)
{
    uint32_t minNodeCapacity = prob.minNodeCount;
    #pragma omp parallel for schedule(dynamic)
    for(uint32_t fid = 0; fid < prob.nr_field; fid += 1)
    {
	//if((fid % 8) != thread_idx) continue;
	const std::vector<std::pair<uint32_t, double>>& instancevec = prob.feat2sortinstances[fid];
        std::vector<Meta> metas = metas0;

	// Instance sort descending order
	for(uint32_t i = 0; i < instancevec.size(); i += 1)
	{
            const std::pair<uint32_t, double> & inst = instancevec[i];
	    uint32_t instid = inst.first;
	    double instval = inst.second;

            Location const &location = instlocations[instid];
            uint32_t tnode_idx = location.tnode_idx;
            if(tnodes[tnode_idx].shrinked)
                  continue;

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

    std::vector<Location> locations(nr_instance);
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
        locations[i].r = R[i];

    for(uint32_t d = 0, offset = 1; d < max_depth; ++d, offset *= 2)
    {
        uint32_t const nr_leaf = static_cast<uint32_t>(pow(2, d));
        std::vector<Meta> metas0(nr_leaf);

        for(uint32_t i = 0; i < nr_instance; ++i)
        {
            Location &location = locations[i];
            if(tnodes[location.tnode_idx].shrinked)
                  continue;

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

        /*
        //for(uint32_t t = 0; t < nr_thread; ++t)
        for(uint32_t t = 0; t < 8; ++t)
	{
        	std::thread thread_f(scan, std::ref(prob), std::ref(locations), std::ref(tnodes), std::ref(metas0), std::ref(defenders), offset, t);
		thread_f.join();
	}
        */

	scan(prob, locations, tnodes, metas0, defenders, offset, 0);

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
        double const r = locations[i].r;
        uint32_t const tnode_idx = locations[i].tnode_idx;
	tnodes[tnode_idx].capacity += 1;
        tmp[tnode_idx].first += r;
        tmp[tnode_idx].second += fabs(r)*(2-fabs(r));
        //tmp[tnode_idx].second += fabs(r)*(1-fabs(r));
    }

    std::map<uint32_t, std::string> curfid2feat = prob.id2featmap;
    for(uint32_t tnode_idx = 1; tnode_idx < max_tnodes; ++tnode_idx)
    {
        double a, b;
        std::tie(a, b) = tmp[tnode_idx];
        tnodes[tnode_idx].gamma = (b <= 1e-12)? 0 : static_cast<double>(a/b);

	//std::cout << "Node: " << tnode_idx << "\tCapacity=" << tnodes[tnode_idx].capacity << "\tScore=" << tnodes[tnode_idx].gamma << endl;
	//std::cout << "Node: " << tnode_idx << "\tfid=" << tnodes[tnode_idx].feature << "\tFeature=" << curfid2feat[tnodes[tnode_idx].feature] << "\tthreshold=" << tnodes[tnode_idx].threshold << endl << endl;
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

void GBDT::fit(Problem &Tr, Problem &Va)
{
    bias = calc_bias(Tr.Y);
    double learningrate = Tr.learningrate;
    //double learningrate = 1;

    std::vector<double> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    Timer timer;
    printf("iter     time    tr_loss    va_loss\n");
    for(uint32_t t = 0; t < trees.size(); ++t)
    {
        timer.tic();

        std::vector<double> const &Y = Tr.Y;
        std::vector<double> R(Tr.nr_instance), F1(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
            //R[i] = static_cast<double>(Y[i]/(1+exp(Y[i]*F_Tr[i])));
            R[i] = static_cast<double>(2*Y[i]/(1+exp(2*Y[i]*F_Tr[i])));

        trees[t].init(this->max_depth);
        trees[t].fit(Tr, R, F1);

        double Tr_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Tr_loss)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += learningrate * F1[i];
            Tr_loss += log(1+exp(-Y[i]*F_Tr[i]));
        }
        Tr_loss /= static_cast<double>(Tr.nr_instance);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Va.nr_instance; ++i)
        {
            std::map<uint32_t, double> x = Va.inst2features[i];
            std::pair<uint32_t, double> res = trees[t].predict(x);
            F_Va[i] += learningrate * res.second;
        }

        double Va_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Va_loss)
        for(uint32_t i = 0; i < Va.nr_instance; ++i) 
            Va_loss += log(1+exp(-Va.Y[i]*F_Va[i]));
        Va_loss /= static_cast<double>(Va.nr_instance);

        printf("%4d %8.1f %10.5f %10.5f\n", t, timer.toc(), Tr_loss, Va_loss);
        fflush(stdout);
    }
}

double GBDT::predict(std::map<uint32_t, double>& x) 
{
    double s = bias;
    for(uint32_t i = 0; i < trees.size(); i += 1)
    {
	CART& tree = trees[i];
        std::pair<uint32_t, double> res = tree.predict(x);
        s += res.second;
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
