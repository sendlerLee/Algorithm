#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"


struct trainer_option
{
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.01), w_beta(0.1), w_l1(0.01), w_l2(0.04),
               v_alpha(0.01), v_beta(0.1), v_l1(0.01), v_l2(0.04), 
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
    virtual void run_task(vector<string>& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void train(int y, const vector<Node>& x);
private:
    ftrl_model* pModel;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    bool k0;
    bool k1;
    bool force_v_sparse;
};


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

void ftrl_trainer::run_task(vector<string>& dataBuffer)
{
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i]);
        train(sample.y, sample.x);
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
void ftrl_trainer::train(int y, const vector<Node>& x)
{
    int xLen = x.size();
    unordered_map<string,ftrl_model_unit*> theta;
    for(int i = 0; i < xLen; ++i)
    {
        for(int j = i + 1; j < xLen; ++j)
        {
            const string& index1 = x[i].feature + "," + x[j].field;
            const string& index2 = x[j].feature + "," + x[i].field;
            theta.insert(make_pair(index1, pModel->getOrInitModelUnit(index1)));
            theta.insert(make_pair(index2, pModel->getOrInitModelUnit(index2)));
        }
    }
    double p = pModel->predict(x, theta);
    double mult = y * (1 / (1 + exp(-p * y)) - 1);
    //cout << p << " " << mult << endl;
    //update w_n, w_z
    for(int i = 0; i < xLen; ++i)
    {
        //update v_n, v_z
        for(int j = i + 1; j < xLen; ++j)
        {
            ftrl_model_unit& mu1 = *(theta[x[i].feature + "," + x[j].field]);
            ftrl_model_unit& mu2 = *(theta[x[j].feature + "," + x[i].field]);
            const double& xi = x[i].value;
            const double& xj = x[j].value;
            for(int f = 0; f < pModel->factor_num; ++f)
            {
                mu1.mtx.lock();
                double& vif1 = mu1.vi[f];
                double& v_nif1 = mu1.v_ni[f];
                double& v_zif1 = mu1.v_zi[f];
                double v_gif1 = v_l2 * vif1 + mult * mu2.vi[f] * xi * xj;
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
                double v_gif2 = v_l2 * vif2 + mult * mu1.vi[f] * xi * xj;
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
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
