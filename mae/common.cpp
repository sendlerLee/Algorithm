#include <stdexcept>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <omp.h>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <fstream>

#include "common.h"
using namespace std;

#define nullptr 0

void read_train_data(Problem& TrProb, std::string const &path)
{
	std::string line;
	std::ifstream inputfile(path);
	while(getline(inputfile, line)) {
		std::istringstream iss(line);
		std::string instance;
		double gap, w, y;
		iss >> instance >> gap;

		TrProb.Gap.push_back(gap);
		y = gap;
		//y = log(gap + 1.0);
		w = 1.0;

		uint32_t instid = static_cast<uint32_t>(TrProb.instidmap.size());
		TrProb.instidmap[instid] = instance;
		TrProb.inst2features.push_back(std::map<uint32_t, double>());

		TrProb.Y.push_back(y);
		TrProb.W.push_back(w);

		std::string feature;
		while (iss) {
			iss >> feature;
			uint32_t findex = static_cast<uint32_t>(feature.find_first_of(":", 0));

			std::string feat = feature.substr(0, findex);
			if (feat.size() == 0) continue;

			uint32_t fid = -1;
			if(TrProb.feat2idmap.find(feat) == TrProb.feat2idmap.end())
			{
				fid = static_cast<uint32_t>(TrProb.feat2idmap.size());
				TrProb.feat2idmap[feat] = fid;
				TrProb.id2featmap[fid] = feat;
				TrProb.feat2sortinstances.push_back(std::vector<std::pair<uint32_t, double>>());
			}
			fid = TrProb.feat2idmap[feat];
			if(TrProb.inst2features[instid].count(fid) > 0) continue;

			double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			TrProb.feat2sortinstances[fid].push_back(std::pair<uint32_t, double>(instid, fval));
			TrProb.inst2features[instid][fid] = fval;
		}
	}

	TrProb.nr_instance = static_cast<uint32_t>(TrProb.inst2features.size());
	TrProb.nr_field = static_cast<uint32_t>(TrProb.feat2sortinstances.size());

	for(uint32_t fid = 0; fid < TrProb.nr_field; fid += 1)
	{ 
		std::vector<std::pair<uint32_t, double>>& instancevec = TrProb.feat2sortinstances[fid];

		std::sort(instancevec.begin(), instancevec.end(), sort_by_v());
		TrProb.feat2sortinstances[fid] = instancevec;
			
	}
}

void read_test_data(Problem& TrProb, Problem& VaProb, std::string const &path)
{
	std::string line;
	std::ifstream inputfile(path);
	while(getline(inputfile, line)) {
		std::istringstream iss(line);
		std::string instance;
		double gap, y;
		iss >> instance >> gap;
		VaProb.Gap.push_back(gap);

		y = gap;
		//y = log(gap + 1.0);
		VaProb.Y.push_back(y);
		
		uint32_t instid = static_cast<uint32_t>(VaProb.instidmap.size());
		VaProb.instidmap[instid] = instance;
		VaProb.inst2features.push_back(std::map<uint32_t, double>());

		std::string feature;
		while (iss) {
			iss >> feature;
			uint32_t findex = static_cast<uint32_t>(feature.find_first_of(":", 0));

			std::string feat = feature.substr(0, findex);
			if (feat.size() == 0) continue;

			uint32_t fid = -1;
			if(TrProb.feat2idmap.count(feat) == 0) continue;
			fid = TrProb.feat2idmap[feat];

			if(VaProb.inst2features[instid].count(fid) > 0) continue;

			double fval = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

			//VaProb.feat2sortinstances.push_back(std::vector<std::pair<uint32_t, double>>());
			//VaProb.feat2sortinstances[fid].push_back(std::pair<uint32_t, double>(instid, fval));
			VaProb.inst2features[instid][fid] = fval;
		}
	}

	VaProb.nr_instance = static_cast<uint32_t>(VaProb.inst2features.size());
	VaProb.nr_field = TrProb.nr_field;
}


std::vector<std::string> argv_to_args(int const argc, char const * const * const argv)
{
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}
