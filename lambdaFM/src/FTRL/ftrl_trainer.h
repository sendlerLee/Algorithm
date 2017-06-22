#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include <algorithm>
#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"


struct trainer_option
{
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
               v_alpha(0.05), v_beta(1.0), v_l1(0.1), v_l2(5.0), 
               threads_num(1), b_init(false), force_v_sparse(false) {}
    string model_path, init_m_path;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num;
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
    return a.second < b.second;
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
    for(int k = 0; k < samples.size(); k++) {
        const vector<pair<string, double> >& x = samples[k].x;
        ftrl_model_unit* thetaBias = pModel->getOrInitModelUnitBias();
        map<string,ftrl_model_unit*> theta;
        int xLen = x.size();
        for(int i = 0; i < xLen; ++i)
        {
            const string& index = x[i].first;
            theta.insert(make_pair(index, pModel->getOrInitModelUnit(index)));
        }
        //update w via FTRL
        for(int i = 0; i <= xLen; ++i)
        {
            ftrl_model_unit& mu = i < xLen ? *(theta[x[i].first]) : *thetaBias;
            if((i < xLen && k1) || (i == xLen && k0))
            {
                mu.mtx.lock();
                if(fabs(mu.w_zi) <= w_l1)
                {
                    mu.wi = 0.0;
                }
                else
                {
                    if(force_v_sparse && mu.w_ni > 0 && 0.0 == mu.wi)
                    {
                        mu.reinit_vi(pModel->init_mean, pModel->init_stdev);
                    }
                    mu.wi = (-1) *
                        (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                        (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);
                }
                mu.mtx.unlock();
            }
        }
        //update v via FTRL
        for(int i = 0; i < xLen; ++i)
        {
            ftrl_model_unit& mu = *(theta[x[i].first]);
            for(int f = 0; f < pModel->factor_num; ++f)
            {
                mu.mtx.lock();
                double& vif = mu.vi[f];
                double& v_nif = mu.v_ni[f];
                double& v_zif = mu.v_zi[f];
                if(v_nif > 0)
                {
                    if(force_v_sparse && 0.0 == mu.wi)
                    {
                        vif = 0.0;
                    }
                    else if(fabs(v_zif) <= v_l1)
                    {
                        vif = 0.0;
                    }
                    else
                    {
                        vif = (-1) *
                            (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                            (v_zif - utils::sgn(v_zif) * v_l1);
                    }
                }
                mu.mtx.unlock();
            }
        }
        vector<double> sum(pModel->factor_num);
        double bias = thetaBias->wi;
        double p = pModel->predict(x, bias, theta, sum);
        estimateY[k] = make_pair(k,p);
        factorSum[k] = sum;
    }

    sort(estimateY.begin(), estimateY.end(), mysortfunc);
    vector<vector<double> > weights(samples.size(),vector<double>(samples.size(),0));
    swapChange(samples,estimateY,weights);

    map<string,pair<double,double> > features;
    for(int m = 0; m < estimateY.size(); m++) {
        int indexM = estimateY[m].first;
        double wxM = estimateY[m].second;
        int yM = samples[indexM].y;
        vector<pair<string, double> > featureM = samples[indexM].x;
        features.clear();
        for(int i = 0; i < featureM.size(); i++) {
            features.insert(make_pair(featureM[i].first,make_pair(featureM[i].second,0.0)));
        }

        for(int n = 0; n < estimateY.size(); n++) {
            int indexN = estimateY[n].first;
            double wxN = estimateY[n].second;
            int yN = samples[indexN].y;
            if(yM == yN) continue;
            
            vector<pair<string, double> > featureN = samples[indexN].x;
            for(int i = 0; i < featureN.size(); i++) {
                string index = featureN[i].first;
                double value = featureN[i].second;
                if(features.count(index) > 0) {
                    features[index] = make_pair(features[index].first,value);    
                }else{
                    features.insert(make_pair(index,make_pair(0.0,value)));
                }
            }
            features.insert(make_pair("bias",make_pair(1.0,1.0)));
            int sign = yM > yN ? 1 : -1;
            double delta_NDCG = sign * abs(weights[m][n]);
            double lambda = -1 / (1 + exp(wxM - wxN));
 
            //update w_n, w_z
            for(map<string,pair<double,double> >::iterator iter = features.begin(); iter != features.end(); ++iter)
            {
                string index = iter -> first;
                pair<double, double> value = iter -> second;
                ftrl_model_unit& mu = (index != "bias") ? *(pModel->getOrInitModelUnit(index)) : *(pModel->getOrInitModelUnitBias());
                double xi = (index != "bias") ? (value.first - value.second) : 1.0;
                if((index != "bias" && k1) || (index == "bias" && k0))
                {
                    mu.mtx.lock();
                    double w_gi = lambda * delta_NDCG * xi;
                    double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
                    mu.w_zi += w_gi - w_si * mu.wi;
                    mu.w_ni += w_gi * w_gi;
                    mu.mtx.unlock();
                }
            }
            //update v_n, v_z
            for(map<string,pair<double,double> >::iterator iter = features.begin(); iter != features.end(); ++iter)
            {
                string index = iter -> first;
                pair<double, double> value = iter -> second;
                ftrl_model_unit& mu = *(pModel->getOrInitModelUnit(index));
                for(int f = 0; f < pModel->factor_num; ++f)
                {
                    mu.mtx.lock();
                    double& vif = mu.vi[f];
                    double& v_nif = mu.v_ni[f];
                    double& v_zif = mu.v_zi[f];
                    double v_gif = lambda * delta_NDCG * ((factorSum[m][f] * value.first - vif * value.first * value.first) - (factorSum[n][f] * value.second - vif * value.second * value.second));
                    double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
                    v_zif += v_gif - v_sif * vif;
                    v_nif += v_gif * v_gif;
                    //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
                    if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
                    {
                        vif = 0.0;
                    }
                    mu.mtx.unlock();
                }
            }

        }
    }    
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
