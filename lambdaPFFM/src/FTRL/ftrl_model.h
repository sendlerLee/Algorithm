#ifndef FTRL_MODEL_H_
#define FTRL_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <cmath>
#include "../Utils/utils.h"

#if defined USEOMP
#include <omp.h>
#endif

using namespace std;

//每一个特征维度的模型单元
class ftrl_model_unit
{
public: double wi;
    double w_ni;
    double w_zi;
    vector<double> vi;
    vector<double> v_ni;
    vector<double> v_zi;
    mutex mtx;
public:
    ftrl_model_unit(int factor_num, double v_mean, double v_stdev)
    {
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num); 
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
            v_ni[f] = 0.0;
            v_zi[f] = 0.0;
        }
    }

    ftrl_model_unit(int factor_num, const vector<string>& modelLineSeg)
    {
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = stod(modelLineSeg[1 + f]);
            v_ni[f] = stod(modelLineSeg[1 + factor_num + f]);
            v_zi[f] = stod(modelLineSeg[1 + 2 * factor_num + f]);
        }
    }

    void reinit_vi(double v_mean, double v_stdev)
    {
        int size = vi.size();
        for(int f = 0; f < size; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
        }
    }
     
    friend inline ostream& operator <<(ostream& os, const ftrl_model_unit& mu)
    {
        for(int f = 0; f < mu.vi.size(); ++f)
        {
            os << " " << mu.vi[f];
        }
        for(int f = 0; f < mu.v_ni.size(); ++f)
        {
            if(abs(mu.v_ni[f]) < 1e-100) {
                os << " " << 0;
            } else {
                os << " " << mu.v_ni[f];
            }
        }
        for(int f = 0; f < mu.v_zi.size(); ++f)
        {
            if(abs(mu.v_zi[f]) < 1e-100) {
                os << " " << 0;
            } else {
                os << " " << mu.v_zi[f];
            }
        }
        return os;
    }
};



class ftrl_model
{
public:
    ftrl_model_unit* muBias;
    unordered_map<string, ftrl_model_unit*> muMap;

    int factor_num;
    int field_num;
    double init_stdev;
    double init_mean;

public:
    ftrl_model(double _factor_num);
    ftrl_model(double _factor_num, double _mean, double _stdev);
    ftrl_model_unit* getOrInitModelUnit(string index);
    ftrl_model_unit* getOrInitModelUnitBias();

    double predict(const vector<Node>& x, unordered_map<string, ftrl_model_unit*>& theta);
    double getScore(const vector<Node>& x, unordered_map<string, ftrl_model_unit*>& theta);
    void outputModel(ofstream& out);
    bool loadModel(ifstream& in);
    void debugPrintModel();

private:
    double get_wi(unordered_map<string, ftrl_model_unit*>& theta, const string& index);
    double get_vif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f);
private:
    mutex mtx;
    mutex mtx_bias;
};


ftrl_model::ftrl_model(double _factor_num)
{
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;
    muBias = NULL;
}

ftrl_model::ftrl_model(double _factor_num, double _mean, double _stdev)
{
    factor_num = _factor_num;
    init_mean = _mean;
    init_stdev = _stdev;
    muBias = NULL;
}


ftrl_model_unit* ftrl_model::getOrInitModelUnit(string index)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.find(index);
    if(iter == muMap.end())
    {
        mtx.lock();
        ftrl_model_unit* pMU = new ftrl_model_unit(factor_num, init_mean, init_stdev);
        muMap.insert(make_pair(index, pMU));
        mtx.unlock();
        return pMU;
    }
    else
    {
        return iter->second;
    }
}


ftrl_model_unit* ftrl_model::getOrInitModelUnitBias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        muBias = new ftrl_model_unit(0, init_mean, init_stdev);
        mtx_bias.unlock();
    }
    return muBias;
}

double ftrl_model::predict(const vector<Node >& x, unordered_map<string, ftrl_model_unit*>& theta)
{
    double result = 0;
    #if defined USEOMP
    #pragma omp parallel for schedule(static) reduction(+: result)
    #endif
    for(int i = 0; i < x.size(); ++i) {
        for(int j = i + 1; j < x.size(); ++j) {
            for(int f = 0; f < factor_num; ++f)
            {
                result += theta[x[i].feature + "," + x[j].field]->vi[f] * x[i].value * theta[x[j].feature + "," + x[i].field]->vi[f] * x[j].value;
            }
        }
    }
    return result;
}

double ftrl_model::getScore(const vector<Node >& x, unordered_map<string, ftrl_model_unit*>& theta)
{
    double result = 0;
    #if defined USEOMP
    #pragma omp parallel for schedule(static) reduction(+: result)
    #endif
    for(int i = 0; i < x.size(); ++i) {
        for(int j = i + 1; j < x.size(); ++j) {
            for(int f = 0; f < factor_num; ++f)
            {
                result += get_vif(theta, x[i].feature + "," + x[j].field, f) * x[i].value * get_vif(theta, x[j].feature + "," + x[i].field, f) * x[j].value;
            }
        }
    }
    return result;
}


double ftrl_model::get_wi(unordered_map<string, ftrl_model_unit*>& theta, const string& index)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->wi;
    }
}


double ftrl_model::get_vif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->vi[f];
    }
}


void ftrl_model::outputModel(ofstream& out)
{
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        out << iter->first << *(iter->second) << endl;
    }
}


void ftrl_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        cout << iter->first << " " << *(iter->second) << endl;
    }
}


bool ftrl_model::loadModel(ifstream& in)
{
    string line;
    vector<string> strVec;
    while(getline(in, line))
    {
        strVec.clear();
        utils::splitString(line, ' ', &strVec);
        if(strVec.size() != 3 * factor_num + 1) 
        {
            return false;
        }
        string& index = strVec[0];
        ftrl_model_unit* pMU = new ftrl_model_unit(factor_num, strVec);
        muMap[index] = pMU;
    }
    return true;
}



#endif /*FTRL_MODEL_H_*/
