#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>
#include <iostream>

using namespace std;

struct Node {
    string field;
    string feature;
    double value;
};

class fm_sample
{
public:
    int y;
    string queryId;
    vector<Node> x;
    fm_sample(const string& line);
};

#endif /*FM_SAMPLE_H_*/
