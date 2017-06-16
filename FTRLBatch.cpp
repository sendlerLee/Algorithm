#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <sys/resource.h>
#include <time.h> 
#include <algorithm>
#include <chrono>
#include <random> 
#include <numeric>
#include <omp.h>
#include <unordered_map>

using namespace std;

class Timer {
public:
    Timer();

    void reset();

    void tic();

    float toc();

    float get();

private:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;
};


Timer::Timer() {
    reset();
}

void Timer::reset() {
    begin = std::chrono::high_resolution_clock::now();
    duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(begin - begin);
}

void Timer::tic() {
    begin = std::chrono::high_resolution_clock::now();
}

float Timer::toc() {
    duration += std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now() - begin);
    return (float) duration.count() / 1000;
}

float Timer::get() {
    float time = toc();
    tic();
    return time;
}


struct Option {
    Option() : alpha(0.5), beta(1), lambda_1(0.01), lambda_2(0.01), algorithm(3), nr_iter(10), auto_pos_repeat(false) {}

    std::string Tr_path, Va_path, FeatWeight_path;
    float alpha, beta, lambda_1, lambda_2;
    int algorithm;
    int nr_iter;
    bool auto_pos_repeat;
};

Option opt;

struct sort_by_v_ascend {
    bool operator()(std::pair<uint32_t, float> const lhs, std::pair<uint32_t, float> const rhs) {
        return lhs.second < rhs.second;
    }
};

double medianw(std::vector<float> const &Y, std::vector<float> const &W) {
    vector<std::pair<uint32_t, float>> instscorevec(Y.size());
    for (uint32_t i = 0; i < Y.size(); ++i)
        instscorevec[i] = pair<uint32_t, float>(i, Y[i]);
    std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v_ascend());

    double halfsum = 0.5 * std::accumulate(W.begin(), W.end(), 0.0);
    double tempsum = 0.0;
    for (int i = 0; i < instscorevec.size(); i += 1) {
        uint32_t idx = instscorevec[i].first;
        double m = instscorevec[i].second;
        double w = W[idx];
        tempsum += w;
        if (tempsum >= halfsum) return m;
    }
}

struct Model {
    int nr_field;

    std::unordered_map<std::string, int> feat2idmap;

    std::vector<float> W;

    std::vector<float> N;
    std::vector<float> Z;

    float F0;
};

struct SampleSet {
    SampleSet() {}

    int nr_instance;

    std::map<int, std::string> id2instance;

    std::vector<vector<pair<int, float>>> X;
    std::vector<float> Y;
    std::vector<float> F;
};


std::string train_help() {
    return std::string(
            "usage: FTRLProximal [<options>] <train_path> <validation_path> <validation_output_path>\n"
                    "\n"
                    "options:\n"
                    "-i <nr_iter>: set the number of iteration\n"
                    "-alg <algorithm>: 1: Gaussian; 2: AdaBoost; 3: Bernoulli; 4: Poisson; 5: Laplace; 6:MAPE; default 3 \n"
                    "-a <alpha>: set alpha\n"
                    "-b <beta>: set beta\n"
                    "-l1 <lambda_1>: L1 Reg lambda_1\n"
                    "-l2 <lambda_2>: L2 Reg lambda_2\n");
}

Option parse_option(std::vector<std::string> const &args) {
    int const argc = static_cast<int>(args.size());

    if (argc == 0)
        throw std::invalid_argument(train_help());

    int i = 0;
    for (; i < argc; ++i) {
        if (args[i].compare("-i") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.nr_iter = stoi(args[++i]);
        } else if (args[i].compare("-alg") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.algorithm = stoi(args[++i]);
        } else if (args[i].compare("-a") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.alpha = stod(args[++i]);
        } else if (args[i].compare("-b") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.beta = stod(args[++i]);
        } else if (args[i].compare("-l1") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.lambda_1 = stod(args[++i]);
        } else if (args[i].compare("-l2") == 0) {
            if (i == argc - 1)
                throw std::invalid_argument("invalid command");
            opt.lambda_2 = stod(args[++i]);
        } else if (args[i].compare("-apr") == 0) {
            opt.auto_pos_repeat = true;
        } else
            break;
    }

    if (i != argc - 3)
        throw std::invalid_argument("invalid command2");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.FeatWeight_path = args[i++];

    return opt;
}

void writeWeightFile(Model &model, SampleSet &Tr, SampleSet &Va, int iter) {
    ofstream outfile(opt.FeatWeight_path + "." + to_string((long long)iter));
    for (auto it = model.feat2idmap.begin(); it != model.feat2idmap.end(); ++it) {
        string feat = it->first;
        int j = it->second;
        float w = model.W[j];
        outfile << feat << " " << w << endl;
    }
    outfile.close();

    ofstream va_res_outfile(opt.Va_path + "." + to_string((long long)iter) + ".pred");
    for (int i = 0; i < Va.nr_instance; i += 1) {
        string inst = Va.id2instance[i];
        float pred = Va.F[i];
        va_res_outfile << inst << " " << pred << endl;
    }
    va_res_outfile.close();
}


struct AUC_SortData {
    int pos;
    int neg;
    AUC_SortData(): pos(0), neg(0) {}
    void add(int y) {
        if(y == 0) {
            neg ++;
        }
        else {
            pos ++;
        }
    }
    int sum() const {
        return pos + neg;
    }
};

double calAUC(const vector<float>& F, const vector<float>& Y) {
    int poscnt = 0;

    map<float, AUC_SortData> sort_data;
    for (int i = 0; i < F.size(); i += 1) {
        float f = F[i];

        int yi = (int) std::round(Y[i]);
        if (yi == 1) poscnt += 1;

        sort_data[f].add(yi);
    }
    int negcnt = F.size() - poscnt;

    int rank = 0;
    double pos_rank_sum = 0;
    for (auto it = sort_data.begin(); it != sort_data.end(); ++it) {
        int current_sample_num = it->second.sum();
        double rank_val = rank + (1 + current_sample_num) * 0.5;
        pos_rank_sum += it->second.pos * rank_val;
        rank += current_sample_num;
    }

    double auc = (pos_rank_sum - (double)poscnt * (poscnt + 1) * 0.5) / poscnt / negcnt;
    return auc;
}

int sign(float x) {
    if (x < 0)
        return -1;
    else
        return 1;
}

std::vector<std::string> argv_to_args(int const argc, char const *const *const argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    return args;
}


void loadTrainInstance(Model &model, SampleSet &Tr, ifstream &inputfile) {
    string line;
    while (getline(inputfile, line)) {
        istringstream iss(line + "\tF0:1");
        string userid;
        float target;
        iss >> userid >> target;

        if (opt.algorithm == 2 || opt.algorithm == 3)
            if (target != 1) target = 0;

        Tr.Y.push_back(target);

        map<string, float> feat2val;
        string feature;
        while (!iss.eof()) {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string feat = feature.substr(0, findex).c_str();
            float x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());
            feat2val[feat] = x;
        }

        Tr.X.push_back(std::vector<pair<int, float>>());
        int i = Tr.X.size() - 1;
        Tr.X[i].resize(feat2val.size(), pair<int, float>());
        Tr.id2instance[i] = userid;

        int j = 0;
        for (map<string, float>::iterator it = feat2val.begin(); it != feat2val.end(); ++it) {
            string feat = it->first;
            float x = it->second;

            int f = -1;
            if (model.feat2idmap.find(feat) == model.feat2idmap.end()) {
                f = model.feat2idmap.size();
                model.feat2idmap[feat] = f;
            }
            f = model.feat2idmap[feat];
            //cout << userid << "\t" << f << "\t" << feat << "\t" << x << endl;

            Tr.X[i][j] = pair<int, float>(f, x);
            j += 1;
        }
        if (Tr.X.size() % 10000 == 0) {
            printf("Loaded %zu lines.\n", Tr.X.size());
        }
    }
    Tr.nr_instance = Tr.X.size();
    model.nr_field = model.feat2idmap.size();
}

void loadTestInstance(Model &model, SampleSet &Tr, SampleSet &Va, ifstream &inputfile) {
    string line;
    while (getline(inputfile, line)) {
        istringstream iss(line + "\tF0:1");
        string userid;
        float target;
        iss >> userid >> target;

        if (opt.algorithm == 2 || opt.algorithm == 3)
            if (target != 1) target = 0;

        map<string, float> feat2val;
        string feature;
        while (!iss.eof()) {
            iss >> feature;
            int findex = feature.find_last_of(":");
            string feat = feature.substr(0, findex).c_str();
            if (model.feat2idmap.count(feat) == 0) continue;

            float x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

            feat2val[feat] = x;
        }

        if (feat2val.size() >= 1) {
            Va.Y.push_back(target);

            Va.X.push_back(std::vector<pair<int, float>>());
            int i = Va.X.size() - 1;
            Va.X[i].resize(feat2val.size(), pair<int, float>());
            Va.id2instance[i] = userid;

            int j = 0;
            for (map<string, float>::iterator it = feat2val.begin(); it != feat2val.end(); ++it) {
                string feat = it->first;
                float x = it->second;
                int f = model.feat2idmap[feat];
                Va.X[i][j] = pair<int, float>(f, x);
                j += 1;
            }
        }
        if (Va.X.size() % 10000 == 0) {
            printf("Loaded %zu lines.\n", Va.X.size());
        }
    }
    Va.nr_instance = Va.X.size();
}

int g_seed = 0;

inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}


int main(int const argc, char const *const *const argv) {
    try {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch (std::invalid_argument const &e) {
        std::cout << e.what();
        return EXIT_FAILURE;
    }

    omp_set_num_threads(50);

    Model model;
    SampleSet Tr, Va;

    ifstream trainfile(opt.Tr_path);
    loadTrainInstance(model, Tr, trainfile);
    cout << "Load Data Finish, numTrainInstance:" << Tr.nr_instance << " numTrainFeature: " << model.nr_field << endl;

    ifstream testfile(opt.Va_path);
    loadTestInstance(model, Tr, Va, testfile);
    cout << "Load Data Finish, numTestInstance:" << Va.nr_instance << endl;

    float alpha = opt.alpha;
    float beta = opt.beta;
    float lambda_1 = opt.lambda_1;
    float lambda_2 = opt.lambda_2;

    Tr.F.resize(Tr.nr_instance, 0);
    model.W.resize(model.nr_field, 0);
    model.N.resize(model.nr_field, 0);
    model.Z.resize(model.nr_field, 0);

    Va.F.resize(Va.nr_instance, 0);

    if (opt.algorithm == 1 || opt.algorithm == 5 || opt.algorithm == 6)
        printf("iter\ttime\tloss\t\tmse\t\tmape\n");
    if (opt.algorithm == 2 || opt.algorithm == 3 || opt.algorithm == 4)
        printf("iter\ttime\tloss\t\tmse\t\tll\n");

    float PosCnt = 0;
    float NegCnt = 0;
    if(opt.algorithm == 2 || opt.algorithm == 3 || opt.algorithm == 4) {
        PosCnt = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0f);
        NegCnt = Tr.Y.size() - PosCnt;
    }

    int pos_repeat = 1;
    if(opt.algorithm == 2 || opt.algorithm == 3) {
        cout << "Pos: " << PosCnt << ", Neg: " << NegCnt << endl;
        if(opt.auto_pos_repeat) {
            double neg_div_pos = NegCnt * 1.0 / PosCnt;
            pos_repeat = (int) ceil(neg_div_pos);
            if(pos_repeat < 1) {
                pos_repeat = 1;
            }
            PosCnt = PosCnt * pos_repeat;
            cout << "pos_repeat: " << pos_repeat << endl;
        }
    }


    //Gaussian
    if (opt.algorithm == 1)
        model.W[model.feat2idmap["F0"]] = std::accumulate(Tr.Y.begin(), Tr.Y.end(), 0.0) * 1.0 / Tr.Y.size();

    //AdaBoost
    if (opt.algorithm == 2) {
        model.F0 = static_cast<float>(log(PosCnt * 1.0 / NegCnt) / 2.0);
    }

    //Bernoulli
    if (opt.algorithm == 3) {
        model.F0 = static_cast<float>(log(PosCnt * 1.0 / NegCnt));
    }

    //Poisson
    if (opt.algorithm == 4) {
        model.F0 = static_cast<float>(log(PosCnt * 1.0 / Tr.Y.size()));
    }

    //Laplace
    if (opt.algorithm == 5) {
        vector<float> xtemp;
        xtemp.resize(Tr.Y.size());
        std::copy(Tr.Y.begin(), Tr.Y.end(), xtemp.begin());
        std::sort(xtemp.begin(), xtemp.end());
        model.F0 = xtemp[int((xtemp.size() - 1) / 2)];
    }

    //MAPE
    if (opt.algorithm == 6) {
        vector<float> W(Tr.Y.size());
        for (uint32_t i = 0; i < Tr.Y.size(); ++i) W[i] = 1.0 / Tr.Y[i];
        model.F0 = medianw(Tr.Y, W);
        //Tr.W[Tr.feat2idmap["F0"]] = Tr.F0;
    }

    vector<int> bevaild(model.nr_field, 0);
    for (int iter = 0; iter < opt.nr_iter; iter += 1) {
        Timer timer;
        timer.reset();
        timer.tic();

        double tr_rmse = 0.0;
        double tr_mape = 0.0;
        for (int i = 0; i < Tr.nr_instance; i += 1) {
            float yi = Tr.Y[i];

            int repeat_num = yi == 1.0 ? pos_repeat : 1;

            for(int repeat_idx = 0; repeat_idx < repeat_num; ++repeat_idx) {

                float Fi = 0;

                for (int t = 0; t < Tr.X[i].size(); t += 1) {
                    pair<int, float> &inst = Tr.X[i][t];
                    int j = inst.first;
                    float x = inst.second;

                    //if (iter != 0 && bevaild[j] == 0) continue;
                    //if (iter == 0 && (rand() % 1000) / 1000.0 > 0.5) continue;

                    if (abs(model.Z[j]) <= lambda_1)
                        model.W[j] = 0.0;
                    else
                        model.W[j] = -(model.Z[j] - sign(model.Z[j]) * lambda_1) / ((beta + sqrt(model.N[j])) / alpha + lambda_2);

                    Fi += model.W[j] * x;
                    //if (iter == 0) bevaild[j] = 1;
                }
                Tr.F[i] = Fi;

                tr_rmse += (yi - Tr.F[i]) * (yi - Tr.F[i]);
                tr_mape += abs(yi - Tr.F[i]) / yi;

                for (int t = 0; t < Tr.X[i].size(); t += 1) {
                    pair<int, float> &inst = Tr.X[i][t];
                    int j = inst.first;
                    float x = inst.second;

                    float g = 0;
                    if (opt.algorithm == 1)
                        g = (Tr.F[i] - yi) * x;

                    if (opt.algorithm == 2)
                        g = -(2 * yi - 1) * exp(-(2 * yi - 1)) * x;

                    if (opt.algorithm == 3) {
                        float pi = 1.0 / (1 + exp(-Tr.F[i]));
                        g = (pi - yi) * x;
                    }

                    if (opt.algorithm == 4)
                        g = (exp(Tr.F[i]) - yi) * x;

                    if (opt.algorithm == 5) {
                        float esp = 10;
                        if (yi - Tr.F[i] > esp) g = -1.0;
                        else if (yi - Tr.F[i] < -esp) g = 1.0;
                        else g = 0;
                    }

                    if (opt.algorithm == 6) {
                        //float esp = 10;
                        float esp = 0;
                        if (yi - Tr.F[i] > esp) g = -1.0 / yi;
                        else if (yi - Tr.F[i] < -esp) g = 1.0 / yi;
                        else g = 0;
                        //g = -(yi - Tr.F[i]) / yi;
                    }

                    float tao = (sqrt(model.N[j] + g * g) - sqrt(model.N[j])) / alpha;
                    model.Z[j] += g - tao * model.W[j];
                    model.N[j] += g * g;
                }
            }
        }
        tr_rmse = sqrt(tr_rmse * 1.0 / Tr.nr_instance);
        tr_mape /= Tr.nr_instance;

        int trimcnt = 0;
#pragma omp parallel for schedule(static) reduction(+: trimcnt)
        for (int j = 0; j < model.W.size(); j += 1) {
            if (fabs(model.W[j]) <= 0)
                trimcnt += 1;
        }
        float sparity = trimcnt * 1.0 / model.nr_field;

        double va_rmse = 0.0;
        double va_mape = 0.0;
        double va_loss = 0.0;
        double va_logloss = 0.0;
#pragma omp parallel for schedule(static) reduction(+: va_rmse, va_mape, va_loss, va_logloss)
        for (int i = 0; i < Va.nr_instance; i += 1) {
            float yi = Va.Y[i];
            Va.F[i] = 0;

            for (vector<pair<int, float>>::iterator it = Va.X[i].begin(); it != Va.X[i].end(); ++it) {
                int j = it->first;
                float x = it->second;
                Va.F[i] += model.W[j] * x;
            }

            float pi = 0;
            if (opt.algorithm == 1)
                va_loss += (Va.F[i] - yi) * (Va.F[i] - yi);

            if (opt.algorithm == 2) {
                va_loss += exp(-(2 * yi - 1) * Va.F[i]);
                pi = 1.0 / (1 + exp(-2 * Va.F[i]));
            }

            if (opt.algorithm == 3) {
                va_loss -= (yi * Va.F[i] - log(1 + exp(Va.F[i])));
                pi = exp(Va.F[i]);
            }

            if (opt.algorithm == 4) {
                va_loss -= (yi * Va.F[i] - exp(Va.F[i]));
                pi = exp(Va.F[i]);
            }

            if (opt.algorithm == 5)
                va_loss += fabs(Va.F[i] - yi);

            if (opt.algorithm == 6)
                va_loss += fabs(Va.F[i] - yi) / yi;

            if (opt.algorithm == 1 || opt.algorithm == 5 || opt.algorithm == 6) {
                va_rmse += (yi - Va.F[i]) * (yi - Va.F[i]);
                va_mape += fabs(yi - Va.F[i]) / yi;
            }

            if (opt.algorithm == 2 || opt.algorithm == 3 || opt.algorithm == 4) {
                if (pi > 0.999) pi = 0.999;
                if (pi < 0.001) pi = 0.001;
                va_rmse += (yi - pi) * (yi - pi);
                va_logloss -= yi * log(pi) + (1 - yi) * log(1 - pi);
            }

        }
        va_loss /= Va.nr_instance;
        va_rmse = sqrt(va_rmse * 1.0 / Va.nr_instance);

        if (opt.algorithm == 1 || opt.algorithm == 5 || opt.algorithm == 6) {
            va_mape /= Va.nr_instance;
            printf("%d\t%.3f\t%.8f\t%.8f\t%.8f", iter, timer.toc(), va_loss, va_rmse, va_mape);
        }

        if (opt.algorithm == 2 || opt.algorithm == 3 || opt.algorithm == 4) {
            va_logloss /= Va.nr_instance;
            double vaauc = calAUC(Va.F, Va.Y);
            printf("%d\t%.3f\t%.8f\t%.8f\t%.8f\t%.8f", iter, timer.toc(), va_loss, va_rmse, va_logloss, vaauc);
        }

        if (opt.lambda_1 > 0) {
            printf("\t%.8f", sparity);
        }
        printf("\n");
        if (iter % 1 == 0 && iter > 0)
            writeWeightFile(model, Tr, Va, iter);
    }

    writeWeightFile(model, Tr, Va, opt.nr_iter);
}
