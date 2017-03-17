#include <vector>
#include <utility>
#include <memory>
#include <mutex>

#include "common.h"

struct TreeNode
{
    TreeNode() : idx(0), feature(-1), threshold(0), gamma(0), shrinked(false), capacity(0) {} 
    uint32_t idx;
    int32_t feature;
    double threshold, gamma;
    bool shrinked;
    uint32_t capacity;
};

class CART 
{
public:
    void init(uint32_t depth)
    {
	max_depth = depth;
	max_tnodes = static_cast<uint32_t>(pow(2, max_depth + 1));
	tnodes.resize(max_tnodes);
        for(uint32_t i = 1; i < max_tnodes; ++i)
            tnodes[i].idx = i;
    }

    CART() : max_depth(0), max_tnodes(0), min_tnode_capacity(0){}

    void fit(Problem &prob, std::vector<double> &R, std::vector<double> &F1, std::vector<std::pair<uint32_t, double> >& tnodeDetail);
    std::pair<uint32_t, double> predict(std::map<uint32_t, double>& x);

    uint32_t max_depth, max_tnodes, min_tnode_capacity;
    std::vector<TreeNode> tnodes;

private:
    static std::mutex mtx;
    static bool verbose;
};

class RandomForest
{
public:
    RandomForest(uint32_t nr_tree, uint32_t nr_depth)
    {
	trees.resize(nr_tree);
	max_depth = nr_depth;
	bias = 0;
    }

    void fit(Problem &Tr, Problem &Va);
    void plrrun(Problem &Tr, Problem &Va);
    void lrrun(Problem &Tr, Problem &Va);
    double predict(std::map<uint32_t, double>& x);
    std::vector<uint32_t> get_indices(std::map<uint32_t, double>& x);
    void write(Problem &prob, std::vector<double> & F_Val,std::string const &path);
    std::string Va_out_path;

private:
    uint32_t max_depth; 
    std::vector<CART> trees;
    double bias;
};
