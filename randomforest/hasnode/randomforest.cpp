#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <omp.h>
#include <stdlib.h>

#include "timer.h"
#include "randomforest.h"

using namespace std;

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
    uint32_t const offset, uint32_t thread_idx)
{
    uint32_t minNodeCapacity = prob.minNodeCount;
    #pragma omp parallel for schedule(dynamic)
    for(uint32_t fid = 0; fid < prob.nr_field; fid += 1)
    {
        srand(100000 * fid + time(NULL));
	uint32_t randval = rand();
        //srand(randval);
        uint32_t ofb = static_cast<uint32_t>(rand() % prob.nr_field);
	//if (ofb * 1.0 > sqrt(1.0 * prob.nr_field)) continue;
	if (ofb * 1.0 > 0.5 * prob.nr_field) continue;

	const std::vector<std::pair<uint32_t, double>>& instancevec = prob.feat2sortinstances[fid];
        std::vector<Meta> metas = metas0;

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

void CART::fit(Problem &prob, std::vector<double> &R, std::vector<double> &F1, std::vector<pair<uint32_t, double> >& tnodeDetail)
{
    uint32_t const nr_field = prob.nr_field;
    uint32_t const nr_instance = prob.nr_instance;

    std::vector<Location> locations(nr_instance);
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < nr_instance; ++i)
    {
        locations[i].r = R[i];
        srand(100000 * i + time(NULL));
	uint32_t randval = rand();
        //srand(randval);
        bool ofb = static_cast<bool>(rand() % 2);
	locations[i].ofb = ofb;
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

	    uint32_t tnode_idx = f + offset;
	    tnodeDetail[tnode_idx] = pair<uint32_t, double>(tnode.feature, tnode.threshold);
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

void RandomForest::write(Problem &prob, vector<double> & F_Val, std::string const &path)
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
	outfile << instance << "\t" << label << "\t" << val << std::endl;

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

void RandomForest::fit(Problem &Tr, Problem &Va)
{
    //bias = calc_bias(Tr.Y);
    bias = 0;
    std::vector<double> F_Tr(Tr.nr_instance, bias), F_Va(Va.nr_instance, bias);

    map<uint32_t, map<uint32_t, double>> nodeprecision;

    Timer timer;
    printf("iter     time    tr_loss    va_loss\n");
    for(uint32_t t = 0; t < trees.size(); ++t)
    {
        uint32_t const nr_node = static_cast<uint32_t>(pow(2, this->max_depth));
    	std::vector<double> const &Y = Tr.Y;
    	std::vector<double> R(Y), F1(Tr.nr_instance);
    	std::vector<pair<uint32_t, double> > tnodeDetail(nr_node);

        timer.tic();

        uint32_t treedepth = 1 + (t / 10);
        if(treedepth > this->max_depth) treedepth = this->max_depth;
        ///trees[t].init(treedepth);
        
        trees[t].init(this->max_depth);
        trees[t].fit(Tr, R, F1, tnodeDetail);

        double Tr_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Tr_loss)
        for(uint32_t i = 0; i < Tr.nr_instance; ++i) 
        {
            F_Tr[i] += F1[i];
            Tr_loss += log(1+exp(-Y[i]*F_Tr[i] * 1.0 / (t + 1)));
        }
        Tr_loss /= static_cast<double>(Tr.nr_instance);

        uint32_t const nr_leaf = static_cast<uint32_t>(pow(2, this->max_depth + 1));
    	std::vector<double> testInstanceNode(Va.nr_instance);
	vector<uint32_t> testNodeInstance(nr_leaf + 1, 0);
	vector<uint32_t> testNodePrecision(nr_leaf + 1, 0);

        #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < Va.nr_instance; ++i)
        {
            std::map<uint32_t, double> x = Va.inst2features[i];
            std::pair<uint32_t, double> res = trees[t].predict(x);
            F_Va[i] += res.second;

	    testInstanceNode[i] = res.first;
        }

	uint32_t testBkgCnt = 0;
	uint32_t testPosCnt = 0;
        for(uint32_t i = 0; i < Va.nr_instance; ++i)
	{
	    uint32_t tnode = testInstanceNode[i];
	    testNodeInstance[tnode] += 1;
	    testBkgCnt += 1;
	    if(Va.Y[i] == 1) 
	    {
		testNodePrecision[tnode] += 1;
		testPosCnt += 1;
	    }
	}

	uint32_t trainBkgCnt = 0;
	uint32_t trainPosCnt = 0;
	vector<uint32_t> nodeInstance(nr_leaf + 1, 0);
	vector<uint32_t> nodePrecision(nr_leaf + 1, 0);
        for(uint32_t i = 0; i < Tr.nr_instance; ++i)
        {
            std::map<uint32_t, double> x = Tr.inst2features[i];
            std::pair<uint32_t, double> res = trees[t].predict(x);

	    uint32_t tnode = res.first;
	    nodeInstance[tnode] += 1;
	    trainBkgCnt += 1;
	    if(Tr.Y[i] == 1) 
	    {
		nodePrecision[tnode] += 1;
		trainPosCnt += 1;
	    }
        }

        for(uint32_t i = 0; i < nr_leaf; ++i)
	{
		if (nodeInstance[i] == 0) continue;
		uint32_t cur_node = i;
		string predictrule = "";
		while(cur_node != 1)
		{
			uint32_t pre_node = static_cast<uint32_t>(cur_node / 2);
			uint32_t pre_left = cur_node % 2;
			string label = " +";
			if (pre_left == 0) label = " -";

			cout << pre_node << " " << Tr.id2featmap[tnodeDetail[pre_node].first] << " " << label << " " << tnodeDetail[pre_node].second << endl;
			predictrule = label + Tr.id2featmap[tnodeDetail[pre_node].first] + predictrule;
			cur_node = pre_node;
		}

		double testposrate = 0;
		double testlift = 0;
		double testbkgrate = testPosCnt * 1.0 / testBkgCnt;
		if (testNodeInstance[i] != 0) 
		{
			testposrate = testNodePrecision[i] * 1.0 / testNodeInstance[i];
			testlift = testposrate * 1.0 / (testposrate + testbkgrate);
		}

		double trainposrate = nodePrecision[i] * 1.0 / nodeInstance[i];
		double trainbkgrate = trainPosCnt * 1.0 / trainBkgCnt;
		double trainlift = trainposrate * 1.0 / (trainposrate + trainbkgrate);

		cout << "Tree: " << t << "\tNode: " << i << "\t" << nodePrecision[i] << "\t" << nodeInstance[i] << "\t" << trainposrate << "\t" << trainbkgrate << "\t" << trainlift << "\t"  << testNodePrecision[i] << "\t" << testNodeInstance[i] << "\t" << testposrate << "\t" << testbkgrate << "\t" << testlift << "\t" << predictrule << endl;

		cout << endl << endl;
		//cout << "Tree: " << t << "\tNode: " << i << "\t" << nodePrecision[i] << "\t" << nodeInstance[i] << "\t" << nodePrecision[i] * 1.0 / nodeInstance[i] << "\t" << predictrule << endl << endl;
	}

        double Va_loss = 0;
        #pragma omp parallel for schedule(static) reduction(+: Va_loss)
        for(uint32_t i = 0; i < Va.nr_instance; ++i) 
            Va_loss += log(1+exp(-Va.Y[i]*F_Va[i] * 1.0 / (t + 1)));
        Va_loss /= static_cast<double>(Va.nr_instance);

        printf("%4d %8.1f %10.5f %10.5f\n", t, timer.toc(), Tr_loss, Va_loss);
	cout << "=======================" << endl << endl;
    //write(Va, F_Va, Va_out_path);
        fflush(stdout);
    }
    write(Tr, F_Tr, "train.result.tmp");
    write(Va, F_Va, Va_out_path);
}

double RandomForest::predict(std::map<uint32_t, double>& x) 
{
    double s = bias;
    for(uint32_t i = 0; i < trees.size(); i += 1)
    {
        std::pair<uint32_t, double> res = trees[i].predict(x);
        s += res.second;
    }
    return s;
}

/*
std::vector<uint32_t> RandomForest::get_node_precision(std::map<uint32_t, double>& x) 
{
    for(uint32_t i = 0; i < Va.nr_instance; ++i)
    {
	std::map<uint32_t, double> x = Va.inst2features[i];
    	uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

	std::vector<uint32_t> indices(nr_tree);
	for(uint32_t t = 0; t < nr_tree; ++t)
		indices[t] = trees[t].predict(x).first;
   }

}
*/

std::vector<uint32_t> RandomForest::get_indices(std::map<uint32_t, double>& x) 
{
    uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

    std::vector<uint32_t> indices(nr_tree);
    for(uint32_t t = 0; t < nr_tree; ++t)
        indices[t] = trees[t].predict(x).first;
    return indices;
}

void RandomForest::plrrun(Problem& Tr, Problem& Va)
{
	cout << "Add New Feature......feat cnt: " << Tr.id2featmap.size() << endl;
	uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

	vector<map<uint32_t, double> > trainPosDataFeature;
	vector<map<uint32_t, double> > trainNegDataFeature;
	map<uint32_t, uint32_t> rffeatmap;

	for(uint32_t i = 0; i < Tr.nr_instance; ++i)
	{
		//std::map<uint32_t, double> x = Tr.inst2features[i];
		std::map<uint32_t, double> x;
		for(uint32_t t = 0; t < nr_tree; ++t)
		{
			std::pair<uint32_t, double> res;
			uint32_t tnode_idx = 1;
			for(uint32_t d = 0; d <= trees[t].max_depth; ++d)
			{
				TreeNode const &tnode = trees[t].tnodes[tnode_idx];
				if(tnode.feature == -1)
					res = std::make_pair(tnode.idx, tnode.gamma);

				if(x.count(tnode.feature) == 0)
					tnode_idx = tnode_idx*2;
				else if(x[tnode.feature] < tnode.threshold)
					tnode_idx = tnode_idx*2;
				else
					tnode_idx = tnode_idx*2+1;
			}

			uint32_t fid = res.first + 1000000 + t * (trees[t].max_tnodes + 1);
			double fval = 1;
			
			x[fid] = fval;
                	if(Tr.Y[i] == 1) trainPosDataFeature.push_back(x);
			else trainNegDataFeature.push_back(x);

			rffeatmap[fid] = 1;
		}
	}

	cout << "End Add New Feature.....feat cnt: " << Tr.id2featmap.size() << endl;

    	map<uint32_t, double> g_featweightmap;
	for(map<uint32_t, uint32_t>::iterator it = rffeatmap.begin(); it != rffeatmap.end(); ++it) 
	{
		uint32_t fid = it->first;
		g_featweightmap[fid] = 0;
	}

	//double nr_lr = 0.000005;
	//double nr_reg = 0.0001;

	double nr_lr = 0.00001;
	double nr_reg = 0.00001;
	uint32_t nr_iter = 100;
	
	uint32_t poscnt = 0;
	uint32_t negcnt = 0;
	vector<map<uint32_t, double> > VaDataFeature;
	for(uint32_t i = 0; i < Va.nr_instance; ++i)
	{
		//std::map<uint32_t, double> x = Tr.inst2features[i];
		std::map<uint32_t, double> x;
		for(uint32_t t = 0; t < nr_tree; ++t)
		{
			std::pair<uint32_t, double> res;
			uint32_t tnode_idx = 1;
			for(uint32_t d = 0; d <= trees[t].max_depth; ++d)
			{
				TreeNode const &tnode = trees[t].tnodes[tnode_idx];
				if(tnode.feature == -1)
					res = std::make_pair(tnode.idx, tnode.gamma);

				if(x.count(tnode.feature) == 0)
					tnode_idx = tnode_idx*2;
				else if(x[tnode.feature] < tnode.threshold)
					tnode_idx = tnode_idx*2;
				else
					tnode_idx = tnode_idx*2+1;
			}

			uint32_t fid = res.first + 1000000 + t * (trees[t].max_tnodes + 1);
			if (rffeatmap.count(fid) == 0) continue;

			double fval = 1;
			x[fid] = fval;
			VaDataFeature.push_back(x);
		}
		if(Va.Y[i] == 1) poscnt += 1;
		else negcnt += 1;
	}

        uint32_t SampleCnt = 5;
	for (uint32_t iter = 0; iter < nr_iter; iter += 1) 
	{
		uint32_t negidx = 0;
        	for (uint32_t idx = 0; idx < trainPosDataFeature.size();  ++idx) 
		{
            		map<uint32_t, double>& posfeaturedict = trainPosDataFeature[idx];

			double wxi = 0.0;
			for (map<uint32_t, double>::iterator fit = posfeaturedict.begin(); fit != posfeaturedict.end(); ++fit) 
			{
				uint32_t fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}

			for (uint32_t s = 0; s < SampleCnt; s += 1)
			{
				srand(100000 * iter + idx * 1000 + s * 10 + time(NULL));
				uint32_t randval = rand();
				srand(randval + time(NULL));
				negidx = rand() % trainNegDataFeature.size();
				
				double wxj = 0.0;
            			map<uint32_t, double>& negfeaturedict = trainNegDataFeature[negidx];
            			map<uint32_t, double> unionfeaturedict = posfeaturedict;
				for (map<uint32_t, double>::iterator fit = negfeaturedict.begin(); fit != negfeaturedict.end(); ++fit) 
				{
					uint32_t fid = fit->first;
					double fval = fit->second;
					wxj += g_featweightmap[fid] * fval;

					if (posfeaturedict.count(fid) > 0) unionfeaturedict[fid] -= fval;
					else unionfeaturedict[fid] = 0 - fval;
				}

				double gradient = - 1.0 / (1 + exp(wxi - wxj));

				for (map<uint32_t, double>::iterator fit = unionfeaturedict.begin(); fit != unionfeaturedict.end(); ++fit) 
				{
					uint32_t fid = fit->first;
					double fval = fit->second;
					g_featweightmap[fid] -= nr_lr * (gradient * fval + nr_reg * g_featweightmap[fid]);
				}
			}
		}
	
		vector<std::pair<uint32_t, double>> instscorevec(Va.nr_instance);

		#pragma omp parallel for schedule(dynamic) 
		for(uint32_t i = 0; i < VaDataFeature.size(); ++i) 
		{
			double wxi = 0.0;
			map<uint32_t, double>& featuredict = VaDataFeature[i];
			for (map<uint32_t, double>::iterator fit = featuredict.begin(); fit != featuredict.end(); ++fit) 
			{
				uint32_t fid = fit->first;
				double fval = fit->second;
				wxi += g_featweightmap[fid] * fval;
			}
	
			instscorevec[i] = pair<uint32_t, double>(i, wxi);
		}
	
		std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v());
	
		vector<uint32_t> pvcnt;
		vector<uint32_t> covcnt;
		double score = 0;
		double avgidx = 0;
		uint32_t startpoint = 1000;
		uint32_t predictposcnt = 0;
		uint32_t belownegcnt = negcnt;
		std::ofstream outfile("TempResult");
		for(uint32_t i = 0; i < instscorevec.size(); ++i)
		{
			uint32_t instid = instscorevec[i].first;
			string instance = Va.instidmap[instid];
	
			double val = instscorevec[i].second;
			double label = Va.Y[instid];
			outfile << instance << "\t" << label << "\t" << val << std::endl;
	
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
			}
		}
		pvcnt.push_back(instscorevec.size()); 
		covcnt.push_back(predictposcnt);
		outfile.close();
	
		uint32_t instsz = instscorevec.size();
		cout << "Iter: " << iter << " avgidx: " << avgidx * 1.0 / instsz << " " << instscorevec.size() << " IDX: " << avgidx * 1.0 / instsz << endl;
		for (uint32_t i = 0; i < pvcnt.size(); i += 1)
			cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;
	
		double auc = score * 1.0 / poscnt;
		cout << "AUC: " << auc << endl << endl;
	}
}

void RandomForest::lrrun(Problem& Tr, Problem& Va)
{
	cout << "Add New Feature......feat cnt: " << Tr.id2featmap.size() << endl;
	uint32_t const nr_tree = static_cast<uint32_t>(trees.size());

	for(uint32_t i = 0; i < Tr.nr_instance; ++i)
	{
		std::map<uint32_t, double> x = Tr.inst2features[i];
		for(uint32_t t = 0; t < nr_tree; ++t)
		{
			std::pair<uint32_t, double> res;
			uint32_t tnode_idx = 1;
			for(uint32_t d = 0; d <= trees[t].max_depth; ++d)
			{
				TreeNode const &tnode = trees[t].tnodes[tnode_idx];
				if(tnode.feature == -1)
					res = std::make_pair(tnode.idx, tnode.gamma);

				if(x.count(tnode.feature) == 0)
					tnode_idx = tnode_idx*2;
				else if(x[tnode.feature] < tnode.threshold)
					tnode_idx = tnode_idx*2;
				else
					tnode_idx = tnode_idx*2+1;
			}

			uint32_t fid = res.first + 1000000 + t * (trees[t].max_tnodes + 1);
			double fval = 1;
			
			Tr.inst2features[i][fid] = fval;
			Tr.id2featmap[fid] = "1";
		}
	}

	cout << "End Add New Feature.....feat cnt: " << Tr.id2featmap.size() << endl;

    	map<uint32_t, double> featweightmap;
	for(map<uint32_t, string>::iterator it = Tr.id2featmap.begin(); it != Tr.id2featmap.end(); ++it) 
	{
		uint32_t fid = it->first;
		featweightmap[fid] = 0;
	}

	//double nr_lr = 0.00002;
	//double nr_reg = 0.00001;

	double nr_lr = 0.0001;
	double nr_reg = 0.00001;
	uint32_t nr_iter = 100;
	
	uint32_t poscnt = 0;
	uint32_t negcnt = 0;
	for(uint32_t i = 0; i < Va.nr_instance; ++i)
	{
		std::map<uint32_t, double> x = Va.inst2features[i];
		for(uint32_t t = 0; t < nr_tree; ++t)
		{
			std::pair<uint32_t, double> res;
			uint32_t tnode_idx = 1;
			for(uint32_t d = 0; d <= trees[t].max_depth; ++d)
			{
				TreeNode const &tnode = trees[t].tnodes[tnode_idx];
				if(tnode.feature == -1)
					res = std::make_pair(tnode.idx, tnode.gamma);

				if(x.count(tnode.feature) == 0)
					tnode_idx = tnode_idx*2;
				else if(x[tnode.feature] < tnode.threshold)
					tnode_idx = tnode_idx*2;
				else
					tnode_idx = tnode_idx*2+1;
			}

			uint32_t fid = res.first + 1000000 + t * (trees[t].max_tnodes + 1);
			if (Tr.id2featmap.count(fid) == 0) continue;
			//if (Va.id2featmap.count(fid) == 0) continue;
			double fval = 1;
			Va.inst2features[i][fid] = fval;
		}
		if(Va.Y[i] == 1) poscnt += 1;
		else negcnt += 1;
	}

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

		/*
		double llh = 0.0;
        	#pragma omp parallel for schedule(static) reduction(+: llh)
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
		*/

		//uint32_t poscnt = 0;
		vector<std::pair<uint32_t, double>> instscorevec(Va.nr_instance);

		#pragma omp parallel for schedule(dynamic) 
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
	
			/*
			uint32_t const nr_tree = static_cast<uint32_t>(trees.size());
			std::map<uint32_t, double> x = Va.inst2features[i];

			for(uint32_t t = 0; t < nr_tree; ++t)
			{
				std::pair<uint32_t, double> res;
				uint32_t tnode_idx = 1;
				for(uint32_t d = 0; d <= trees[t].max_depth; ++d)
				{
					TreeNode const &tnode = trees[t].tnodes[tnode_idx];
					if(tnode.feature == -1)
						res = std::make_pair(tnode.idx, tnode.gamma);

					if(x.count(tnode.feature) == 0)
						tnode_idx = tnode_idx*2;
					else if(x[tnode.feature] < tnode.threshold)
						tnode_idx = tnode_idx*2;
					else
						tnode_idx = tnode_idx*2+1;
				}

				//std::pair<uint32_t, double> res = trees[t].predict(x);
				uint32_t fid = res.first + 1000000 + t * (trees[t].max_tnodes + 1);
				double fval = 1;
				if (Tr.id2featmap.count(fid) == 0) continue;
				wxi += featweightmap[fid] * fval;
			}
	
			if(Va.Y[i] == 1) poscnt += 1;
			*/
			instscorevec[i] = pair<uint32_t, double>(i, wxi);
		}
	
		std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v());
	
		vector<uint32_t> pvcnt;
		vector<uint32_t> covcnt;
		double score = 0;
		double avgidx = 0;
		uint32_t startpoint = 1000;
		uint32_t predictposcnt = 0;
                uint32_t belownegcnt = negcnt;
		std::ofstream outfile("TempResult");
		for(uint32_t i = 0; i < instscorevec.size(); ++i)
		{
			uint32_t instid = instscorevec[i].first;
			string instance = Va.instidmap[instid];
	
			double val = instscorevec[i].second;
			double label = Va.Y[instid];
			outfile << instance << "\t" << label << "\t" << val << std::endl;
	
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
			}
		}
		pvcnt.push_back(instscorevec.size()); 
		covcnt.push_back(predictposcnt);
		outfile.close();
	
		uint32_t instsz = instscorevec.size();
		cout << "Iter: " << iter << " avgidx: " << avgidx * 1.0 / instsz << " " << instscorevec.size() << " IDX: " << avgidx * 1.0 / instsz << endl;
		for (uint32_t i = 0; i < pvcnt.size(); i += 1)
			cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;
	
		double auc = score * 1.0 / poscnt;
		cout << "AUC: " << auc << endl << endl;
	}
}
