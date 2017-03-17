#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <algorithm>

#include "common.h"
#include "timer.h"
#include "gbdt.h"

using namespace std;

struct Option
{
    Option() : nr_tree(30), nr_thread(5), nr_minnodecnt(50), learningrate(0.05) {}
    std::string Tr_path, Va_path, Va_out_path;
    uint32_t nr_depth, nr_minnodecnt, nr_tree, nr_thread;
    double learningrate;
};

std::string train_help()
{
    return std::string(
"usage: gbdt [<options>] <train_path> <validation_path> <validation_output_path>\n"
"\n"
"options:\n"
"-d <depth>: set the maximum depth of a tree\n"
"-m <minNodeCount>: set the minimum number of node\n"
"-s <nr_thread>: set the maximum number of threads\n"
"-l <learningrate>: set the learning rate\n"
"-t <nr_tree>: set the number of trees\n");
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
	else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.learningrate = std::stof(args[++i]);
        }

        else if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_tree = std::stoi(args[++i]);
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw std::invalid_argument("invalid command");
            opt.nr_thread = std::stoi(args[++i]);
        }
        else
        {
            break;
        }
    }

    if(i != argc-3)
        throw std::invalid_argument("invalid command");

    opt.Tr_path = args[i++];
    opt.Va_path = args[i++];
    opt.Va_out_path = args[i++];

    return opt;
}

void write(Problem &prob, GBDT &gbdt, std::string const &path)
{
    uint32_t poscnt = 0;
    vector<std::pair<uint32_t, double>> instscorevec;
    for(uint32_t instid = 0; instid < prob.nr_instance; ++instid)
    {
	if(prob.Y[instid] == 1) poscnt += 1;
	double val = gbdt.predict(prob.inst2features[instid]);
	instscorevec.push_back(pair<uint32_t, double>(instid, val));
    }
    
    std::sort(instscorevec.begin(), instscorevec.end(), sort_by_v());

    vector<uint32_t> pvcnt;
    vector<uint32_t> covcnt;
    double score = 0;
    double avgidx = 0;
    uint32_t startpoint = 1000;
    uint32_t predictposcnt = 0;
    uint32_t belowposcnt = poscnt;
    std::ofstream outfile(path);
    for(uint32_t i = 0; i < instscorevec.size(); ++i)
    {
	uint32_t instid = instscorevec[i].first;
	string instance = prob.instidmap[instid];
	std::vector<uint32_t> indices = gbdt.get_indices(prob.inst2features[instid]);

	double val = instscorevec[i].second;
	double label = prob.Y[instid];
	outfile << instance << "\t" << label << "\t" << val << "\t" << indices[0] << std::endl;

	if (label == 1) {
		score += static_cast<double>((instscorevec.size() - i - belowposcnt) * 1.0 / (poscnt));
		belowposcnt -= 1;
		avgidx += i * 1.0 / poscnt;
		predictposcnt += 1;
	}
	if (i == startpoint) {
		pvcnt.push_back(i);
		covcnt.push_back(predictposcnt);
		startpoint *= 2;
	}
     }
     pvcnt.push_back(instscorevec.size());
     covcnt.push_back(predictposcnt);
     outfile.close();

     cout << "avgidx: " << avgidx * 1.0 / poscnt << " " << instscorevec.size() << " IDX: " << avgidx * 1.0 / (instscorevec.size()) << endl;
     for (uint32_t i = 0; i < pvcnt.size(); i += 1)
	cout << covcnt[i] << "\t" << pvcnt[i] << "\t" << covcnt[i] * 1.0 / pvcnt[i] << endl;

     double auc = score * 1.0 / (instscorevec.size() - poscnt);
     cout << "AUC: " << auc << endl << endl;
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
    Tr.learningrate = opt.learningrate;
    std::cout << "Train nr_instance: " << Tr.nr_instance << " nr_field: " << Tr.nr_field << " nr_minnodecnt: " << Tr.minNodeCount << std::endl;
    std::cout << "done\n" << std::flush;

    std::cout << "Reading test data...\n" << std::flush;
    read_test_data(Tr, Va, opt.Va_path);
    Va.minNodeCount = opt.nr_minnodecnt;
    std::cout << "Test nr_instance: " << Va.nr_instance << " nr_field: " << Va.nr_field << " nr_minnodecnt: " << Va.minNodeCount << std::endl;
    std::cout << "done\n" << std::flush;

    omp_set_num_threads(static_cast<int>(opt.nr_thread));

    GBDT gbdt(opt.nr_tree, opt.nr_depth);
    gbdt.fit(Tr, Va);

    write(Tr, gbdt, "train.result.tmp");
    write(Va, gbdt, opt.Va_out_path);

    return EXIT_SUCCESS;
}
