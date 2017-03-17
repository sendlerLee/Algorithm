#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <cmath>

#include <pmmintrin.h>

struct sort_by_v
{
	bool operator() (std::pair<uint32_t, double> const lhs, std::pair<uint32_t, double> const rhs)
	{
		return lhs.second > rhs.second;
	}
};

struct Problem
{
    Problem() : nr_instance(0), nr_field(0) {}
    Problem(uint32_t const nr_instance, uint32_t const nr_field) 
        : nr_instance(nr_instance), nr_field(nr_field),
          Y(nr_instance) {}

    uint32_t nr_instance, nr_field;
    uint32_t minNodeCount;
    double learningrate;

    std::map<std::string, uint32_t> feat2idmap;
    std::map<uint32_t, std::string> id2featmap;
    std::vector<std::vector<std::pair<uint32_t, double>>> feat2sortinstances;

    std::map<uint32_t, std::string> instidmap;
    std::vector<std::map<uint32_t, double>> inst2features;

    std::vector<double> Y;
};

void read_train_data(Problem& VaProb, std::string const &path);
void read_test_data(Problem& TrProb, Problem& VaProb, std::string const &path);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

#endif // _COMMON_H_
