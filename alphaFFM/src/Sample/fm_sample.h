#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>

using namespace std;

const string spliter = " ";
const string innerSpliter = ":";

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


fm_sample::fm_sample(const string& line) 
{
    this->x.clear();
    size_t posb = line.find_first_not_of(spliter, 0);
    size_t pose = line.find_first_of(spliter, posb);
    int label = atoi(line.substr(posb, pose-posb).c_str());
    this->y = label > 0 ? 1 : -1;
    posb = line.find_first_not_of(spliter, pose);
    if(posb == string::npos)
    {
        cout << "wrong line input\n" << line << endl;
        throw "wrong line input";
    }
    pose = line.find_first_of(spliter, posb);
    queryId = line.substr(posb, pose-posb);
    string field;
    string feature;
    double value;
    while(pose < line.size())
    {
        posb = line.find_first_not_of(spliter, pose);
        if(posb == string::npos)
        {
            break;
        }
        pose = line.find_first_of(innerSpliter, posb);
        if(pose == string::npos)
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        field = line.substr(posb, pose-posb);

        posb = pose + 1;
        pose = line.find_first_of(innerSpliter, posb);
        if(pose == string::npos)
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        feature = line.substr(posb, pose-posb);

        posb = pose + 1;
        if(posb >= line.size())
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        pose = line.find_first_of(spliter, posb);
        value = stod(line.substr(posb, pose-posb));
        if(value != 0)
        {
            this->x.push_back({field,feature,value});
        }
    }
}


#endif /*FM_SAMPLE_H_*/
