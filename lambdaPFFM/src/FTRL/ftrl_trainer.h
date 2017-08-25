#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include <algorithm>
#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"
#if defined USEOMP
#include <omp.h>
#endif

struct trainer_option
{
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.01), w_beta(0.01), w_l1(0.1), w_l2(0.04),
               v_alpha(0.01), v_beta(0.01), v_l1(0.1), v_l2(0.04), field_num(1),
               threads_num(1), b_init(false), force_v_sparse(false) {}
    string model_path, init_m_path;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num, field_num;
    bool k0, k1, b_init, force_v_sparse;
    
    void parse_option(const vector<string>& args) 
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0) 
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                vector<string> strVec;
                string tmpStr = args[++i];
                utils::splitString(tmpStr, ',', &strVec);
                if(strVec.size() != 3)
                    throw invalid_argument("invalid command\n");
                k0 = 0 == stoi(strVec[0]) ? false : true;
                k1 = 0 == stoi(strVec[1]) ? false : true;
                factor_num = stoi(strVec[2]);
            }
            else if(args[i].compare("-init_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_beta = stod(args[++i]);
            }
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-v_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-v_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_beta = stod(args[++i]);
            }
            else if(args[i].compare("-v_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-v_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                v_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_m_path = args[++i];
                b_init = true; //if im field exits , that means b_init = true !
            }
            else if(args[i].compare("-fvs") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                int fvs = stoi(args[++i]);
                force_v_sparse = (1 == fvs) ? true : false;
            }
            else if(args[i].compare("-f") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                field_num = stoi(args[++i]); 
            }   
            else
            {
                throw invalid_argument("invalid command\n");
                break;
            }
        }
    }

};


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);
    virtual void run_task(vector<vector<fm_sample>>& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void train(const vector<fm_sample>& samples);
private:
    ftrl_model* pModel;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    bool k0;
    bool k1;
    bool force_v_sparse;
};

bool mysortfunc(const pair<int, double>& a, const pair<int, double>& b)
{
    return a.second > b.second;
}

double getIdealDCG(vector<int> rel) {
    sort(rel.begin(),rel.end(), greater<int>());
    double dcg = 0.0;
    for(int i = 1; i <= rel.size(); i++) {
        dcg += (pow(2.0,rel[i - 1]) - 1) / log2(i + 1);
    }
    return dcg;
}

void swapChange(const vector<fm_sample>& samples,const vector<pair<int, double> >& estimates,vector<vector<double> >& weights) {
    vector<int> rel;
    for(int i = 0; i < estimates.size(); i++) {
        int index = estimates[i].first;
        rel.push_back(samples[index].y);
    }
    double idealDcg = getIdealDCG(rel);
    
    if(idealDcg > 0) {
        for(int i = 1; i <= rel.size(); i++) {
            for(int j = i + 1; j <= rel.size(); j++) {
                weights[i - 1][j - 1] = weights[j - 1][i - 1] = (1.0 / log2(i + 1) - 1.0 / log2 (j + 1)) * (pow(2.0, rel[i - 1]) - pow(2.0, rel[j - 1])) / idealDcg;
            }
        }
    }
}

ftrl_trainer::ftrl_trainer(const trainer_option& opt)
{
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    v_alpha = opt.v_alpha;
    v_beta = opt.v_beta;
    v_l1 = opt.v_l1;
    v_l2 = opt.v_l2;
    k0 = opt.k0;
    k1 = opt.k1;
    force_v_sparse = opt.force_v_sparse;
    pModel = new ftrl_model(opt.factor_num, opt.init_mean, opt.init_stdev);
}

void ftrl_trainer::run_task(vector<vector<fm_sample> >& dataBuffer)
{
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        train(dataBuffer[i]);
    }
}


bool ftrl_trainer::loadModel(ifstream& in)
{
    return pModel->loadModel(in);
}


void ftrl_trainer::outputModel(ofstream& out)
{
    return pModel->outputModel(out);
}


//输入一个样本，更新参数
void ftrl_trainer::train(const vector<fm_sample>& samples)
{
    vector<pair<int,double> > estimateY(samples.size());
    map<int,vector<double> > factorSum;
    unordered_map<string,ftrl_model_unit*> theta;
    for(int k = 0; k < samples.size(); k++) {
        const vector<Node >& x = samples[k].x;
        int xLen = x.size();
        for(int i = 0; i < xLen; ++i)
        {
            for(int j = i + 1; j < xLen; ++j) {
                const string& index1 = x[i].feature + "," + x[j].field;
                const string& index2 = x[j].feature + "," + x[i].field;
                theta.insert(make_pair(index1, pModel->getOrInitModelUnit(index1)));
                theta.insert(make_pair(index2, pModel->getOrInitModelUnit(index2)));
            }
        }
        double p = pModel->predict(x,theta);
        estimateY[k] = make_pair(k,p);
    }

    sort(estimateY.begin(), estimateY.end(), mysortfunc);
    vector<vector<double> > weights(samples.size(),vector<double>(samples.size(),0));
    swapChange(samples,estimateY,weights);

    for(int m = 0; m < estimateY.size(); m++) {
        int indexM = estimateY[m].first;
        double wxM = estimateY[m].second;
        int yM = samples[indexM].y;
        vector<Node> featureM = samples[indexM].x;
        unordered_map<string,double> tempFeatures;
        for(int i = 0; i < featureM.size(); i++) {
            for(int j = i + 1; j < featureM.size(); j++) {
                tempFeatures.insert(make_pair(featureM[i].feature + "," + featureM[j].field + ";" + featureM[j].feature + "," + featureM[i].field,featureM[i].value * featureM[j].value));
            }   
        }

        for(int n = m + 1; n < estimateY.size(); n++) {
            int indexN = estimateY[n].first;
            double wxN = estimateY[n].second;
            int yN = samples[indexN].y;
            if(yM == yN) continue;
            
            unordered_map<string,double> features(tempFeatures);
            vector<Node> featureN = samples[indexN].x;
            
            for(int i = 0; i < featureN.size(); i++) {
                for(int j = i + 1; j < featureN.size(); j++) {
                    string index = featureN[i].feature + "," + featureN[j].field + ";" + featureN[j].feature + "," + featureN[i].field;
                    double value = featureN[i].value * featureN[j].value;
                    auto it = features.find(index);
                    if(it != features.end()) {
                        it->second = it->second - value;    
                    } else {
                        features.insert(make_pair(index, -1 * value));
                    }
                }
            }
            int sign = yM > yN ? 1 : -1;
            double delta_NDCG = sign * abs(weights[m][n]);
            double lambda = -1.0 / (1 + exp(sign * (wxM - wxN))) * delta_NDCG;
 
            //update v_n, v_z
            for(unordered_map<string,double>::iterator iter = features.begin(); iter != features.end(); ++iter)
            {
                string index = iter -> first;
                double value = iter -> second;
                int iPos = index.find_first_of(";",0);
                string key1 = index.substr(0,iPos);
                string key2 = index.substr(iPos + 1,index.size() - iPos);
                ftrl_model_unit& mu1 = *(theta[key1]);
                ftrl_model_unit& mu2 = *(theta[key2]);
                for(int f = 0; f < pModel->factor_num; ++f)
                {
                    mu1.mtx.lock();
                    double& vif1 = mu1.vi[f];
                    double& v_nif1 = mu1.v_ni[f];
                    double& v_zif1 = mu1.v_zi[f];
                    double v_gif1 = v_l2 * vif1 + lambda * mu2.vi[f] * value;
                    double v_sif1 = 1 / v_alpha * (sqrt(v_nif1 + v_gif1 * v_gif1) - sqrt(v_nif1));
                    v_zif1 += v_gif1 - v_sif1 * vif1;
                    v_nif1 += v_gif1 * v_gif1;
                    if(fabs(v_zif1) <= v_l1)
                    {
                        vif1 = 0.0;
                    }
                    else
                    {
                        vif1 = (-1) *
                            (1 / (v_l2 + (v_beta + sqrt(v_nif1)) / v_alpha)) *
                            (v_zif1 - utils::sgn(v_zif1) * v_l1);
                    }
                    mu1.mtx.unlock();
                    mu2.mtx.lock();
                    double& vif2 = mu2.vi[f];
                    double& v_nif2 = mu2.v_ni[f];
                    double& v_zif2 = mu2.v_zi[f];
                    double v_gif2 = v_l2 * vif2 + lambda * mu1.vi[f] * value;
                    double v_sif2 = 1 / v_alpha * (sqrt(v_nif2 + v_gif2 * v_gif2) - sqrt(v_nif2));
                    v_zif2 += v_gif2 - v_sif2 * vif2;
                    v_nif2 += v_gif2 * v_gif2;
                    if(fabs(v_zif2) <= v_l1)
                    {
                        vif2 = 0.0;
                    }
                    else
                    {
                        vif2 = (-1) *
                            (1 / (v_l2 + (v_beta + sqrt(v_nif2)) / v_alpha)) *
                            (v_zif2 - utils::sgn(v_zif2) * v_l1);
                    }
                    mu2.mtx.unlock();
                }
            }
        }
    }    
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
