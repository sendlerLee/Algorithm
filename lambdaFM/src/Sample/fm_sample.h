#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>
#include <iostream>

using namespace std;

class fm_sample
{
public:
    int y;
    string queryId;
    vector<pair<string, double> > x;
    fm_sample(const string& line);
};

#endif /*FM_SAMPLE_H_*/
