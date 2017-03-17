#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <tr1/unordered_map>
using namespace std;

#define hashmap std::tr1::unordered_map

inline bool startswith(const string &src, const string &dest){
    return src.compare(0, dest.size(), dest) == 0;
}

inline bool mysortfunc(int i,int j) {return (i<j);}

int getdomain(const string & url, string &domain){
    size_t pos = url.find("://");
    if (pos == std::string::npos){
        return -1;
    }
    pos += 3;
    size_t start = pos, end = 0;
    end = url.find("/", start);
    if (end == string::npos){
        domain = url.substr(start);
        return 0;
    }

    domain = url.substr(start, end - start);
    return 0;
}


inline bool ifIn(string &elem, set<string> &container){
    bool flag = false;
    if (container.find(elem) != container.end())
        flag = true;
    return flag;
}

void split(std::string& s, std::string delim,std::vector< std::string >* ret)
{
    size_t last = 0;
    size_t index=s.find(delim,last);
    while (index!=std::string::npos)
    {
        ret->push_back(s.substr(last,index-last));
        last=index+delim.size();
        index=s.find(delim,last);
    }
    if (index-last>0)
    {
        ret->push_back(s.substr(last,index-last));
    }
}

void trim(std::string & s){
    s.erase(0, s.find_first_not_of(" /n"));
    s.erase(s.find_last_not_of(' ') + 1);
}

int calc_file_lines(string featurelogname){
    ifstream ifs(featurelogname.c_str());
    string line;
    int linenum = 0;
    while (getline(ifs, line)){
        linenum++;
    }
    ifs.close();
    return linenum;
}

int pos_samples_cnt = 0;
hashmap<string, int> convUserdict;
vector<int> datelist;

hashmap<string, float> url2ctr;

hashmap<string, string> userid2labeldict;
hashmap<string, map<int, float> > userid2featuredict;
hashmap<int, map<string, int> > feature2labelcnt;
map<string, int> label2cnt;

int linecnt = 0;
map<string, int> feature2id;

void process(string featurefilename, string label){
    ifstream ifs(featurefilename.c_str());
    string line, delim("\t");

    while (getline(ifs, line)){
        vector<string> v_lineitem;
        trim(line);
        split(line, delim, &v_lineitem);
        if (v_lineitem.size() != 2) continue;
        string pyid(v_lineitem[0]);

        linecnt++;
        // if (linecnt % 10000 == 0) cout << "linecnt = " << linecnt << endl;
        int loopcnt = 1;
        if (convUserdict.find(pyid) != convUserdict.end()){
            if (pos_samples_cnt < 30000)
                loopcnt = 10;
            if (pos_samples_cnt < 10000)
                loopcnt = 20;
        }

        for (int index = 0; index < loopcnt; ++index){
            hashmap<string, int> featureset;
            vector<string> v_url;
            v_url.reserve(100);
            split(v_lineitem[1], string(""), &v_url);
            if (v_url.size() > 250)
                break;

            for (vector<string>::iterator it = v_url.begin(); it != v_url.end(); ++it){
                vector<string> logtype_url;
                split(*it, "", &logtype_url);
                if (logtype_url.size() < 2) continue;
                string logtype(logtype_url[0]), url(logtype_url[1]);
                if (*(url.end() - 1) == '/') url.erase(url.end() - 1);
                string url_str(logtype + "" + url);
                if (featureset.find(url_str) == featureset.end())
                    featureset[url_str] = 0;
                featureset[url_str] += 1;

                vector<string> urlitem;
                size_t maxitem = 6;
                urlitem.reserve(16);
                split(url, string("/"), &urlitem);
                if (logtype != "bid_unbid")
                    maxitem = 4;
                for (size_t i = 3; i < std::min(urlitem.size(), maxitem); ++i){
                    char newurl[2048] = {0};
                    std::strcat(newurl, logtype.c_str());
                    std::strcat(newurl, "");
                    std::strcat(newurl, urlitem[0].c_str());
                    for (size_t j = 1; j < i; ++j) {
                        if (std::strlen(newurl) + 1 + urlitem[j].size() < 2048){
                            std::strcat(newurl, "/");
                            std::strcat(newurl, urlitem[j].c_str());
                        }
                    }
                    string newurl_str(newurl);
                    if (featureset.find(newurl_str) == featureset.end())
                        featureset[newurl_str] = 0;
                    featureset[newurl_str] += 1;
                }
            }// for (vector<string>::iterator it = v_url.begin(); it != v_url.end(); ++it)

            map<int, float> fidset;
            for (hashmap<string, int>::iterator it = featureset.begin(); it != featureset.end(); ++it){
                string url(it->first);
                if (url2ctr.find(url) == url2ctr.end())    continue;
                if (feature2id.find(url) == feature2id.end())
                    feature2id[url] = feature2id.size();
                int fid = feature2id[url];
                if (feature2labelcnt[fid].find(label) == feature2labelcnt[fid].end())
                    feature2labelcnt[fid][label] = 0;
                feature2labelcnt[fid][label] += 1;
                fidset[fid] = log(it->second + 1.0) * url2ctr[url];
            }
            if (fidset.size() == 0) continue;
            if (label2cnt.find(label) == label2cnt.end())
                label2cnt[label] = 0;
            label2cnt[label] += 1;
            userid2labeldict[pyid] = label;
            userid2featuredict[pyid] = fidset;
        }// for (int index = 0; index < loopcnt; ++index)
    }
    ifs.close();
}

int main(int argc, char *argv[]){
    if (argc < 3){
        cout << "Error!" << endl << argv[0] << " inputid targetWin" << endl;
        return -1;
    }
    string inputid(argv[1]);
    string targetWin(argv[2]);

    string delim(" ");
    vector<string> v_date;
    split(targetWin, delim, &v_date);
    for (vector<string>::iterator it = v_date.begin(); it != v_date.end(); ++it){
        int date = atoi(it->c_str());
        datelist.push_back(date);
    }
    sort(datelist.begin(), datelist.end(), mysortfunc);

    set<int> targetWindows;
    for (vector<int>::iterator it = datelist.begin(); it != datelist.end() - 1; ++it){
        targetWindows.insert(*it);
    }
    cout << "train targetWindows = ";
    for (set<int>::iterator it = targetWindows.begin(); it != targetWindows.end(); ++it){
        cout << " " << *it;
    }
    cout << endl;

    string line;

    string featurefilename = string("conv/conv_")+inputid;
    ifstream ifs_conv(featurefilename.c_str());
    if (ifs_conv){
        while (getline(ifs_conv, line)){
            vector<string> v_pyid_date;
            string delim = "\t";
            trim(line);
            split(line, delim, &v_pyid_date);

            string pyid = v_pyid_date[0];
            convUserdict[pyid] = 1;
        }
    }
    else{
        cout << "Warning: can\'t find file or read data "<< featurefilename << endl;
        return -1;
    }
    ifs_conv.close();
    // cout << "convUserdict.size() ..." << convUserdict.size() << endl;

    pos_samples_cnt = calc_file_lines(string("feature_log/feature_")+inputid+string("_log"));
    int bkgcnt_thrshld = 10;
    if (pos_samples_cnt < 6500)
        bkgcnt_thrshld = 5;
    else if (pos_samples_cnt < 15000)
        bkgcnt_thrshld = 7;
    else if (pos_samples_cnt < 35000)
        bkgcnt_thrshld = 10;
    else
        bkgcnt_thrshld = 12;

    ifstream ifs_nb((string("feature_select/feature_")+inputid+string("_select")).c_str());
    while (getline(ifs_nb, line)){
        vector<string> v_nb;
        trim(line);
        split(line, string(" "), &v_nb);
        if (v_nb.size() != 9)
            continue;

        string url = v_nb[0], ctr = v_nb[8], cnt = v_nb[4];
        if (atoi(cnt.c_str()) <= bkgcnt_thrshld)
            continue;
        url2ctr[url] = atof(ctr.c_str());
    }
    ifs_nb.close();
    cout << "url2ctr.size() ..." << url2ctr.size() << endl;

    process("feature_log/feature_0_log", "0");
    process(string("feature_log/feature_") + inputid + string("_log"), "1");

    int label_pos_thrshld = 3;
    if (pos_samples_cnt < 6000)
        label_pos_thrshld = 2;
    else if (pos_samples_cnt < 10000)
        label_pos_thrshld = 3;
    else if (pos_samples_cnt < 40000)
        label_pos_thrshld = 4;
    else if (pos_samples_cnt < 100000)
        label_pos_thrshld = 5;
    else if (pos_samples_cnt < 200000)
        label_pos_thrshld = 6;
    else
        label_pos_thrshld = 7;

    int fcnt_thrshld = 3;
    if (pos_samples_cnt < 5000)
        fcnt_thrshld = 1;
    else if (pos_samples_cnt < 15000)
        fcnt_thrshld = 2;
    else if (pos_samples_cnt < 25000)
        fcnt_thrshld = 4;
    else if (pos_samples_cnt < 50000)
        fcnt_thrshld = 5;
    else if (pos_samples_cnt < 100000)
        fcnt_thrshld = 6;
    else
        fcnt_thrshld = 7;

    // feature selection
    set<int> validdict;
    cout << "len(feature2labelcnt) = " << feature2labelcnt.size() << endl;
    for (hashmap<int, map<string, int> >::iterator it = feature2labelcnt.begin(); it != feature2labelcnt.end(); ++it){
        int fid = it->first;
        if (feature2labelcnt[fid].find("1") == feature2labelcnt[fid].end()
              || feature2labelcnt[fid]["1"] < label_pos_thrshld){
            continue;
        }
        validdict.insert(fid);
    }

    vector<string> instancelist;
    instancelist.reserve(8192);
    int neg_sample_cnt = 0;
    cout << "label2cnt[1] = " << label2cnt["1"] << endl;
    cout << "len(userid2featuredict) = " << userid2featuredict.size() << endl;
    cout << "len(validdict) = " << validdict.size() << endl;
    int seqno = 0;
    for (hashmap<string, string>::iterator it = userid2labeldict.begin(); it != userid2labeldict.end(); it++){
        string pyid = it->first;
        string label = userid2labeldict[pyid];
        map<int, float> fidset = userid2featuredict[pyid];
        char tmpstr[2048] = {0};
        int feature_count = 0, init_len = 0;
        sprintf(tmpstr, "%d", seqno);
        strcat(tmpstr, " ");
        strcat(tmpstr, label.c_str());
        strcat(tmpstr, " ");
        init_len = strlen(tmpstr);
        for (map<int, float>::iterator fit = fidset.begin(); fit != fidset.end(); ++fit){
            if (validdict.find(fit->first) == validdict.end()) continue;
            char fid_str[8] = {0}, fidset_str[16] = {0};
            sprintf(fid_str, "%d", fit->first);
            sprintf(fidset_str, "%.5f", fit->second);
            // feature += str(fid) + ":" + str(fidset[fid]) + " "
            if (strlen(tmpstr) + strlen(fid_str) + 1 + strlen(fidset_str) + 1 < 2048){
                strcat(tmpstr, fid_str);
                strcat(tmpstr, ":");
                strcat(tmpstr, fidset_str);
                strcat(tmpstr, " ");
                feature_count += 1;
            }
        }
        if (strlen(tmpstr) == init_len) continue;
        char *ptr = tmpstr + std::strlen(tmpstr) - 1;
        while (*ptr == ' '){
            *ptr = 0;
            ptr++;
        }
        string feature(tmpstr);
        if (label == "0"){
            if (neg_sample_cnt > label2cnt["1"] * 1.5) continue;
            if (feature_count < fcnt_thrshld) continue;
            neg_sample_cnt += 1;
        }
        instancelist.push_back(feature);
        seqno++;
    }

    ofstream featurefh((string("feature_file/feature_file_") + inputid).c_str(), ios::out);
    for (map<string, int>::iterator it = feature2id.begin(); it != feature2id.end(); ++it){
        if (validdict.find(it->second) == validdict.end())
            continue;
        featurefh << it->second << " " << it->first << "\n";
    }
    featurefh.close();

    cout << "train instancelist.size() = " << instancelist.size() << endl;
    ofstream trainfh((string("campaign_ext/campaign_ext.train.") + inputid).c_str(), ios::out);
    for (vector<string>::iterator it = instancelist.begin(); it != instancelist.end(); ++it)
        trainfh << *it << "\n";
    trainfh.close();

    return 0;
}

