#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <stdlib.h>

using namespace std;

struct sort_by_v_ascend
{
	bool operator() (std::pair<int, double> const lhs, std::pair<int, double> const rhs)
	{
		return lhs.second < rhs.second;
	}
};

double medianw(std::vector<double> const &Y, std::vector<double> const &W)
{
    vector<std::pair<int, double> > instscorevec(Y.size());
    for(int i = 0; i < Y.size(); ++i)
        instscorevec[i] = pair<int, double>(i, Y[i]);
    std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v_ascend());

    double halfsum = 0.5 * std::accumulate(W.begin(), W.end(), 0.0); 
    double tempsum = 0.0;
    for(int i = 0; i < instscorevec.size(); i += 1)
    {
	int idx = instscorevec[i].first;
	double m = instscorevec[i].second;
	double w = W[idx];
	tempsum += w;
	if (tempsum >= halfsum) return m;
    }
}

int main(int const argc, char const * const * const argv)
{
	std::vector<double> Y;
	std::vector<double> W;
	for (int i = 0; i <= 100; i += 1)
	{
		Y.push_back(i);
		W.push_back(1);
	}
	cout << medianw(Y, W) << endl;;
}
