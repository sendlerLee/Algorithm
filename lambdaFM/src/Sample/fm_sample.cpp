#include "fm_sample.h"

const string spliter = " ";
const string innerSpliter = ":";

fm_sample::fm_sample(const string& line) 
{
    x.clear();
    size_t posb = line.find_first_not_of(spliter, 0);
    size_t pose = line.find_first_of(spliter, posb);
    int label = atoi(line.substr(posb, pose-posb).c_str());
    y = label > 0 ? 1 : 0;
    posb = line.find_first_not_of(spliter, pose);
    if(posb == string::npos)
    {
        cout << "wrong line input\n" << line << endl;
        throw "wrong line input";
    }
    pose = line.find_first_of(spliter, posb);
    queryId = line.substr(posb, pose-posb);
    string key;
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
        key = line.substr(posb, pose-posb);
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
            this->x.push_back(make_pair(key, value));
        }
    }
}

