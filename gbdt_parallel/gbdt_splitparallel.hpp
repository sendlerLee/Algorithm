#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ctime>
#include <cassert>
#include <omp.h>
using namespace std;

struct ForestConfig {    
    int min_children;
    int depth;
    int max_feature;
    int tree_cnt;
    int max_pos;
    float bootstrap;
    float step ;      
    int nthread;
    
    ForestConfig() {
        min_children = 10;
        depth = 8;
        tree_cnt = 50;
        max_feature = -1;        
        max_pos = -1;
        bootstrap = 0;
        step = 0.1;            
        nthread = 1;
    }
};


struct DFeature {
    vector<float> f;
    float y;    
};

struct SFeature {
    vector< pair<int, float> > f;
    float y;
};

struct TNode
{    
    float value;
    float splitval;
    int ind;
    int ch[2];
    float sum_y;
    float sum_sqr_y;
};

const float EPSI = 1e-3;

unsigned long long now_rand = 1;

double get_time() {    
    struct timeval   tp;
    struct timezone  tzp;
    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );	
}

void set_rand_seed(unsigned long long seed)
{
    now_rand = seed;
}

unsigned long long get_rand()
{
    now_rand = ((now_rand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
    return now_rand;
}

inline float sqr(const float &x) {
    return x * x;
}

inline int sign(const float &val) {
    if (val > EPSI) return 1;
    else if (val < -EPSI) return -1;
    else return 0;
}

double tot_parallel_time = 0.0;

class DecisionTree {
    private:    
    struct QNode {
        int nid;                
        int left, right;
        
        QNode() {
            nid = left = right = 0;
        }        
        
        QNode(const int &nid_, const int &left_, const int &right_) {
            nid = nid_;
            left = left_;            
            right = right_;
        }
    };    
    
    struct SplitInfo {
        int bind;
        float bsplit;
        int cnt[2];
        float sum_y[2], sum_sqr_y[2];
        float err;
        
        void update(const SplitInfo &sinfo) {
            bind = sinfo.bind;
            bsplit = sinfo.bsplit;
            cnt[0] = sinfo.cnt[0]; cnt[1] = sinfo.cnt[1];
            sum_y[0] = sinfo.sum_y[0]; sum_y[1] = sinfo.sum_y[1];
            sum_sqr_y[0] = sinfo.sum_sqr_y[0]; sum_sqr_y[1] = sinfo.sum_sqr_y[1];
            err = sinfo.err;
        }
    };
    
    vector<QNode> q;

    vector<SplitInfo> split_infos;
    
    public:
    vector<TNode> tree;        
    
    int min_children;
    int max_depth;
    
    int n; // number of instances
    int m; // number of features
    int nthread; // number of threads
    
    private:
    void init_data() {
        q.reserve(256);
        q.resize(0);
        split_infos.reserve(256);
        split_infos.resize(0);
        tree.reserve(256);
        tree.resize(0);
        
        omp_set_num_threads(nthread);
        
        #pragma omp parallel
        {
            this->nthread = omp_get_num_threads();
        }
        printf("number of thread: %d\n", this->nthread);
    }       
    
    void update_queue() {
        vector<QNode> new_q;
        TNode new_node;
        for (int i = 0; i < q.size(); i++) {
            /*
            printf("nid: %d, left: %d, right: %d\n", q[i].nid, q[i].left, q[i].right);
            printf("bind: %d, bsplit: %f, cnt0: %d, cnt1: %d\n", split_infos[i].bind, 
                split_infos[i].bsplit, split_infos[i].cnt[0], split_infos[i].cnt[1]);
            printf("sum0: %f, sum1: %f, v0: %f, v1: %f\n", split_infos[i].sum_y[0], 
                split_infos[i].sum_y[1], split_infos[i].sum_y[0] / max(1, split_infos[i].cnt[0]),
                split_infos[i].sum_y[1] / max(1, split_infos[i].cnt[1]));
            printf("\n");
            */
            if (split_infos[i].bind >= 0 && split_infos[i].cnt[0] >= min_children && split_infos[i].cnt[1] >= min_children) {
                int ii = q[i].nid;
                tree[ii].ind = split_infos[i].bind;                        
                tree[ii].splitval = split_infos[i].bsplit;                
                tree[ii].ch[0] = tree.size();
                tree[ii].ch[1] = tree.size() + 1;
                
                new_q.push_back(QNode(tree.size(), q[i].left, q[i].left + split_infos[i].cnt[0]));
                new_q.push_back(QNode(tree.size() + 1, q[i].left + split_infos[i].cnt[0], q[i].right));
                                                
                for (int c = 0; c < 2; c++) {
                    new_node.ind = -1;
                    new_node.value = split_infos[i].sum_y[c] / split_infos[i].cnt[c];
                    new_node.sum_y = split_infos[i].sum_y[c];
                    new_node.sum_sqr_y = split_infos[i].sum_sqr_y[c];
                    tree.push_back(new_node);
                }                
            }
        }
        q = new_q;
    }
    
    void expand(int q_ind, vector<DFeature> &features, vector<int> &id_list, 
        vector< pair<float, int> > &tlist) {        
        
        int nid = q[q_ind].nid;
        int left = q[q_ind].left;
        int right = q[q_ind].right;
        //printf("processing qnode %d, left %d, right %d...\n", q_ind, left, right);
        
        float tmp_val = features[id_list[left]].y;
        bool flag = true;
        for (int i = left + 1; i < right; i++) {
            if (tmp_val != features[id_list[i]].y) {
                flag = false;
                break;
            }
        }
        // all the y values are identical
        if (flag) return;
        
        //vector< pair<float, int> > tlist(right - left);
        //vector< pair<float, int> > tlist((right - left) * m);
        int p = m * left;
        for (int j = 0; j < m; j++) {            
            for (int i = left; i < right; i++) {
                tlist[p].first = features[id_list[i]].f[j];
                tlist[p++].second = id_list[i];
            }
        }                               
                        
        SplitInfo gbest;
        gbest.err = 1e100;
        gbest.bind = -1;    
        float sum = tree[nid].sum_y;
        float ss = tree[nid].sum_sqr_y;
        
        #pragma omp parallel
        {                    
            SplitInfo tbest;
            tbest.err = 1e100;
            tbest.bind = -1;
            #pragma omp for schedule(dynamic,1) //(static) //
            for (int fea_ind = 0; fea_ind < m; ++fea_ind) {
                float sum0, ss0, sum1, ss1;                                   
                
                int start_ind = left * m + fea_ind * (right - left);                                
               // printf("start to sort feature %d, st_ind: %d, en_ind: %d, tlist size: %d\n", fea_ind,
                //    start_ind, start_ind + (right - left),tlist.size());
                sort(tlist.begin() + start_ind, tlist.begin() + (start_ind + (right-left)));            
                
                sum0 = ss0 = 0.0;
                int len = right - left;
                for (int i = 0; i + 1 < len; i++) {
                    int id = tlist[i + start_ind].second;
                    sum0 += features[id].y;
                    ss0 += sqr(features[id].y);
                    if (len - i - 1 < min_children) break;
                    if (i + 1 < min_children || sign(tlist[i + start_ind].first - tlist[i + start_ind + 1].first) == 0) continue;                            
                    
                    sum1 = sum - sum0;
                    ss1 = ss - ss0;                               
                    
                    //float v0 = sum0 / (i - left + 1);
                    //float v1 = sum1 / (right - i - 1);
                    
                    float err = ss0 - sum0 * sum0 / (i + 1) + ss1 - sum1 * sum1 / (len - i - 1);
                    //float err = ss0 - 2 * v0 * sum0 + v0 * v0 * (i + 1)
                    //    + ss1 - 2 * v1 * sum1 + v1 * v1 * (tlist.size() - i - 1);
                    
                    if (err < tbest.err) {                        
                        tbest.err = err;
                        tbest.bind = fea_ind;                    
                        tbest.bsplit = (tlist[i + start_ind].first + tlist[i + start_ind + 1].first) / 2;
                        //printf("new err: %f, nbind: %d, nbsplit: %f\n", tbest.err, 
                        //    tbest.bind, tbest.bsplit);
                        tbest.sum_y[0] = sum0; tbest.sum_y[1] = sum1;
                        tbest.sum_sqr_y[0] = ss0; tbest.sum_sqr_y[1] = ss1;
                        tbest.cnt[0] = i + 1; tbest.cnt[1] = len - i - 1;
                    }
                }                
            }
            
            #pragma omp critical
            {
                if (tbest.err < gbest.err)
                    gbest.update(tbest);
            }
        }
        
        //printf("q_ind: %d, gbind: %d, gbsplit: %f, gerr: %f\n", q_ind, gbest.bind, gbest.bsplit,
        //    gbest.err);
        
        split_infos[q_ind].bind = gbest.bind;
                             
        if (gbest.bind >= 0) {
            split_infos[q_ind].update(gbest);            
            
            int cur_ind = left * m + (right - left) * gbest.bind;
                        
            for (int i = left; i < right; i++) {
                id_list[i] = tlist[cur_ind++].second;                
            }                  
        }
    }
    
    public:
    DecisionTree(vector<DFeature> &features, int max_depth, int max_feature, int max_pos, 
        int min_children, float bootstrap, int nthread) {
        
        this->n = features.size();        
        this->m = features.size() > 0 ? features[0].f.size() : 0;
        this->min_children = max(min_children, 1);
        this->max_depth = max_depth;
        this->nthread = nthread ? nthread : 1;
        
        init_data();
        
        vector<int> id_list;
        id_list.reserve(n);
        float sum_y = 0.0;
        float sum_sqr_y = 0.0;
        int tcnt = 0;
        // process bootstrap
        for (int i = 0; i < n; i++) {
            if ((float)get_rand() / RAND_MAX >= bootstrap) {
                id_list.push_back(i);
                sum_y += features[i].y;
                sum_sqr_y += sqr(features[i].y);                
            }             
        }
        
        // add the root node        
        TNode node;
        node.ind = -1;
        node.value = sum_y / (id_list.size() ? id_list.size() : 1);
        node.sum_y = sum_y;
        node.sum_sqr_y = sum_sqr_y;        
        tree.push_back(node);
        
        if (id_list.size() == 0) return;
        
        q.push_back(QNode(0, 0, id_list.size()));  

        vector< pair<float, int> > tlist(id_list.size() * m);
        
        
        // build a dection tree         
        for (int dep = 0; dep < max_depth; dep++) {
            if (q.size() == 0) break;
            //printf("building depth %d...\n", dep);
            
            split_infos.resize(q.size());
            //printf("size of split info: %d\n", split_infos.size());
            if (dep < 3) {
                for (int i = 0; i < q.size(); i++) {
                    this->expand(i, features, id_list, tlist);
                }
            } else {
                double start_time = get_time();
                int nq = q.size();
                
                for (int i = 0; i < nq; ++i) {
                    this->expand(i, features, id_list, tlist);
                }
                tot_parallel_time += get_time() - start_time;
            }
            
            this->update_queue();            
        }    
                              
#ifdef cpp11        
        tree.shrink_to_fit();
#endif     
    }
        
    float predictTree(vector<float> &f) {
        int n = 0;
        while (tree[n].ind >= 0)
        {
            if (f[ tree[n].ind ] <= tree[n].splitval)
                n = tree[n].ch[0];
            else
                n = tree[n].ch[1];
        }
        return tree[n].value;
    }           
};

float cal_rmse(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    float rmse = 0;
    for (int i = 0; i < pred.size(); i++) {
        rmse += sqr(pred[i] - gt[i]);
    }
    rmse = sqrt(rmse / pred.size());
    return rmse;
}

float cal_auc(vector<float> &pred, vector<float> &gt) {
    assert(pred.size() == gt.size());
    vector< pair<float, float> > tv;
    for (int i = 0; i < pred.size(); i++)
        tv.push_back(make_pair(pred[i], -gt[i]));
    sort(tv.begin(), tv.end());
    for (int i = 0; i < tv.size(); i++) 
        tv[i].second = -tv[i].second;
    int pos_cnt = 0, neg_cnt = 0;
    float cor_pair = 0;
    for (int i = 0; i < tv.size(); i++)
        if (tv[i].second > 0.5) {
            pos_cnt++;
            cor_pair += neg_cnt;
        } else {
            neg_cnt++;
        }
    return (neg_cnt > 0 && pos_cnt > 0) ? (cor_pair / pos_cnt / neg_cnt) : 0.0;
}

class BoostedForest {
    public:
    vector<DecisionTree*> trees;
    int depth, max_feature, max_pos, min_children, nthread;
    float bootstrap, step;
    vector<float> cur_vals, ori_vals;
    vector<float> steps;
    vector<DFeature> *val_features_ptr;

    BoostedForest() {
        val_features_ptr = NULL;
        step = 0.1;
        depth = 5;
        max_feature = max_pos = -1;
        min_children = 50;
    }
    
    void set_val_data(vector<DFeature> &data) {
        val_features_ptr = &data;
    }
    
    void buildForest(vector<DFeature> &features, int num_tree, int depth_, int max_feature_, 
        int max_pos_, int min_children_, float bootstrap_, float step_, int nthread_) {        
                
        depth = depth_;
        max_feature = max_feature_;
        max_pos = max_pos_;
        min_children = min_children_;
        bootstrap = bootstrap_;
        step = step_;
        nthread = nthread_;
        if (max_feature < 0) max_feature = int(sqrt(features[0].f.size()) + 1);
        
        cur_vals = vector<float>(features.size());
        ori_vals = vector<float>(features.size());
        for (int i = 0; i < features.size(); i++)
            ori_vals[i] = features[i].y;
        
        vector<float> val_vals;
        vector<float> pred_vals;
        if (val_features_ptr != NULL) {
            vector<DFeature> &val_features = *val_features_ptr;
            pred_vals = vector<float>(val_features.size());
            val_vals = vector<float>(val_features.size());
            for (int i = 0; i < val_features.size(); i++)
                val_vals[i] = val_features[i].y;
        }        
        
        float train_rmse = -1, test_rmse = -1;
        float train_auc = -1, test_auc = -1;        
        
        double start_time = get_time();
        for (int i = 0; i < num_tree; i++)
        {
            for (int j = 0; j < features.size(); j++)
                features[j].y = ori_vals[j] - cur_vals[j];
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);
            
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;            
            }                        
            train_rmse = cal_rmse(cur_vals, ori_vals);
            train_auc = cal_auc(cur_vals, ori_vals);
            
            if (val_features_ptr != NULL) {                
                vector<DFeature> &val_features = *val_features_ptr;
                for (int j = 0; j < val_features.size(); j++) {
                    pred_vals[j] += dt->predictTree(val_features[j].f) * step;                    
                }
                test_rmse = cal_rmse(pred_vals, val_vals);
                test_auc = cal_auc(pred_vals, val_vals);
            }
            
            steps.push_back(step);
            
            printf("iter: %d, train_rmse: %.6lf, test_rmse: %.6lf, tree_size: %d\n", i + 1, train_rmse, test_rmse, dt->tree.size());
            printf("train_auc: %.6lf, test_auc: %.6lf\n", train_auc, test_auc);            
            printf("%.3f seconds passed for training, %.3f seconds in parallel\n", get_time() - start_time, tot_parallel_time);
        }
        
        double train_time = get_time() - start_time;
        FILE *fout = fopen("time.out", "w");
        fprintf(fout, "%.3f\n", train_time);
        fclose(fout);
        
        for (int j = 0; j < features.size(); j++)
            features[j].y = ori_vals[j];
    }

    void buildForest(vector<DFeature> &features, ForestConfig &conf) {
        buildForest(features, conf.tree_cnt, conf.depth, conf.max_feature, conf.max_pos, 
            conf.min_children, conf.bootstrap, conf.step, conf.nthread);
    }
    
    
    void addTree(vector<DFeature> &features) {        
        addTree(features, 1);
    }

    void addTree(vector<DFeature> &features, int treecnt) {        
        for (int j = 0; j < features.size(); j++) {
            ori_vals[j] = features[j].y;            
        }
        while (treecnt--) {
            for (int j = 0; j < features.size(); j++) {                
                features[j].y = ori_vals[j] - cur_vals[j];
            }
            DecisionTree *dt = new DecisionTree(features, depth, max_feature, max_pos, min_children, bootstrap, nthread);
            trees.push_back(dt);                          
            for (int j = 0; j < features.size(); j++) {
                cur_vals[j] += dt->predictTree(features[j].f) * step;
            }
            steps.push_back(step);
        }
        for (int j = 0; j < features.size(); j++) {            
            features[j].y = ori_vals[j];
        }
    }
    
    void set_step(float step_) {
        step = step_;
    }
    
    float predictForest(vector<float> &f) {
        float ret = 0;        
        for (int j = 0; j < trees.size(); j++) {
            ret += trees[j]->predictTree(f) * steps[j];            
        }        
        return ret;
    }
};
