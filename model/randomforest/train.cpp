#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <algorithm>

#include "common.h"
#include "timer.h"
#include "randomforest.h"

using namespace std;

struct Option
{
	Option() : nr_depth(5), nr_minnodecnt(50), nr_tree(30), rffeat(0){}
	std::string Tr_path, Va_path, Va_out_path;
	uint32_t nr_depth, nr_minnodecnt, nr_tree;
	uint32_t rffeat;
};

std::string train_help()
{
    return std::string(
"usage: randomforest [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-t <nr_tree>: set the number of trees\n"
"-d <depth>: set the maximum depth of a tree\n"
"-f <rf feat>: get rf feature\n"
"-m <minNodeCount>: set the minimum number of node\n");
}

Option parse_option(std::vector<std::string> const &args)
{
    uint32_t const argc = static_cast<uint32_t>(args.size());

    if(argc == 0)
        throw std::invalid_argument(train_help());

    Option opt; 

    uint32_t i = 0;
    for(; i < argc; ++i)
    {
        if(args[i].compare("-d") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_depth = std::stoi(args[++i]);
        }
	else if(args[i].compare("-m") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_minnodecnt = std::stoi(args[++i]);
        }
        else if(args[i].compare("-f") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.rffeat = std::stoi(args[++i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_tree = std::stoi(args[++i]);
        }
        else break;
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

int main(int const argc, char const * const * const argv)
{
    Option opt;
    try
    {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e)
    {
        std::cout << e.what();
        return EXIT_FAILURE;
    }

    std::cout << "Reading train data...\n" << std::flush;

    Problem Tr, Va; 
    read_train_data(Tr, opt.Tr_path);
    Tr.minNodeCount = opt.nr_minnodecnt;
    Tr.input_path = opt.Tr_path;
    std::cout << "Train nr_instance: " << Tr.nr_instance << " nr_field: " << Tr.nr_field << " nr_minnodecnt: " << Tr.minNodeCount << std::endl;
    std::cout << "done\n" << std::flush;

    std::cout << "Reading test data...\n" << std::flush;
    read_test_data(Tr, Va, opt.Va_path);
    Va.minNodeCount = opt.nr_minnodecnt;
    Va.input_path = opt.Va_path;
    std::cout << "Test nr_instance: " << Va.nr_instance << " nr_field: " << Va.nr_field << " nr_minnodecnt: " << Va.minNodeCount << std::endl;
    std::cout << "done\n" << std::flush;

    omp_set_num_threads(20);

    RandomForest randomforest(opt.nr_tree, opt.nr_depth);
    randomforest.Va_out_path = opt.Va_out_path;
    randomforest.rffeat = opt.rffeat;
    randomforest.fit(Tr, Va);
    return EXIT_SUCCESS;
}
