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
vector<int> datelist;

hashmap<string, float> url2ctr;
hashmap<string, int> url2fid;

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
        string pyid(v_lineitem[0]), label("0");

        linecnt++;
        // if (linecnt % 10000 == 0) cout << "linecnt = " << linecnt << endl;
        int loopcnt = 1;
        for (int index = 0; index < loopcnt; ++index){
            hashmap<string, int> featureset;
            vector<string> v_url;
            v_url.reserve(100);
            split(v_lineitem[1], "", &v_url);
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
                split(url, "/", &urlitem);
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
                if (url2fid.find(url) == url2fid.end())    continue;
                int fid = url2fid[url];
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
    targetWindows.insert(*(datelist.end() - 1));
    cout << "test targetWindows = ";
    for (set<int>::iterator it = targetWindows.begin(); it != targetWindows.end(); ++it){
        cout << " " << *it;
    }
    cout << endl;

    string line;

    pos_samples_cnt = calc_file_lines(string("feature_log/feature_")+inputid+string("_log"));

    ifstream ifs_fid((string("feature_file/feature_file_")+inputid).c_str());
    while (getline(ifs_fid, line)){
        vector<string> v_fid_url;
        trim(line);
        split(line, string(" "), &v_fid_url);
        if (v_fid_url.size() != 2)
            continue;
        url2fid[v_fid_url[1]] = atoi(v_fid_url[0].c_str());
    }

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
        string delim = " ";
        trim(line);
        split(line, delim, &v_nb);
        if (v_nb.size() != 9) continue;

        string url = v_nb[0], ctr = v_nb[8], cnt = v_nb[4];
        if (atoi(cnt.c_str()) <= bkgcnt_thrshld)
            continue;
        url2ctr[url] = atof(ctr.c_str());
    }
    ifs_nb.close();
    cout << "url2ctr.size() ..." << url2ctr.size() << endl;

    process("feature_log/feature_0_log", "0");
    process(string("feature_log/feature_") + inputid + string("_log"), "1");

    vector<string> instancelist;
    instancelist.reserve(8192);
    int index = 0;
    for (hashmap<string, string>::iterator it = userid2labeldict.begin(); it != userid2labeldict.end(); it++){
        string pyid = it->first;
        string label = userid2labeldict[pyid];
        map<int, float> fidset = userid2featuredict[pyid];
        char tmpstr[2048] = {0};
        int init_len = 0;
        sprintf(tmpstr, "%d", index++);
        strcat(tmpstr, " ");
        strcat(tmpstr, label.c_str());
        strcat(tmpstr, " ");
        init_len = strlen(tmpstr);
        for (map<int, float>::iterator fit = fidset.begin(); fit != fidset.end(); ++fit){
            char fid_str[8] = {0}, fidset_str[16] = {0};
            sprintf(fid_str, "%d", fit->first);
            sprintf(fidset_str, "%.5f", fit->second);
            // feature += str(fid) + ":" + str(fidset[fid]) + " "
            if (strlen(tmpstr) + strlen(fid_str) + 1 + strlen(fidset_str) + 1 < 2048){
                strcat(tmpstr, fid_str);
                strcat(tmpstr, ":");
                strcat(tmpstr, fidset_str);
                strcat(tmpstr, " ");
            }
        }
        if (strlen(tmpstr) == init_len) continue;
        char *ptr = tmpstr + std::strlen(tmpstr) - 1;
        while (*ptr == ' '){
            *ptr = 0;
            ptr++;
        }
        string feature(tmpstr);
        instancelist.push_back(feature);
    }

    cout << "test instancelist.size() = " << instancelist.size() << endl;
    ofstream testfh((string("campaign_ext/campaign_ext.test.") + inputid).c_str(), ios::out);
    for (vector<string>::iterator it = instancelist.begin(); it != instancelist.end(); ++it)
        testfh << *it << "\n";
    testfh.close();

    return 0;
}

