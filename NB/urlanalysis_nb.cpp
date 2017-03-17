#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <tr1/unordered_map>
#include <algorithm>
#include <time.h>
using namespace std;

#define hashmap std::tr1::unordered_map

double total_time = 0.;

inline bool startswith(const string &src, const string &dest){
    return src.compare(0, dest.size(), dest) == 0;
}

inline int getdomain(const string & url, string &domain){
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

inline bool ifIn(const string &elem, const set<string> &container){
    bool flag = false;
    if (container.find(elem) != container.end())
        flag = true;
    return flag;
}

inline void split(const std::string& s, const std::string &delim,std::vector< std::string >* ret)
{
    clock_t start, end;
    start = clock();

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
    end = clock();

    total_time += (double)(end - start);
}

int main(int argc, char *argv[]){
    if (argc < 3){
        cout << "Error!" << endl << argv[0] << " dirpath inputid" << endl;
        return -1;
    }
    string dirpath(argv[1]);
    string inputid(argv[2]);

    set<string> cheat_domain;
    set<string> shield_domain;
    ifstream ifs_cheat("cheat_domain.txt");
    ifstream ifs_shield("shield_domain.txt");

    string ch_domain;
    string sh_domain;
    while (getline(ifs_cheat, ch_domain)){
        cheat_domain.insert(ch_domain);
    }
    while (getline(ifs_shield, sh_domain)){
        shield_domain.insert(sh_domain);
    }

    int poscnt = 0, negcnt = 0;
    map<string, int> feature2poscnt;
    map<string, int> feature2negcnt;
    hashmap<string, int> feature2bkgcnt;

    map<string, int>::iterator neg_it;
    map<string, int>::iterator pos_it;
    hashmap<string, int>::iterator bkg_it;
/*
    for (set<string>::iterator it=shield_domain.begin();it!= shield_domain.end();++it)
        cout << *it<<endl;
    return -1;
*/
    string line;
    ifstream ifs_pos_feature_log((dirpath + "/feature_" + inputid + "_log").c_str());
    
    while (getline(ifs_pos_feature_log, line)){
        istringstream iss;
        iss.str(line);

        string pyid, urllist;
        iss >> pyid;
        iss >> urllist;

        set<string> featureset;
        string delim("");
        vector<string> v_url;
        v_url.reserve(100);

        split(urllist, delim, &v_url);
        for (vector<string>::iterator it = v_url.begin(); it < v_url.end(); ++it){
            vector<string> logtype_url;
            split(*it, "", &logtype_url);
            if (logtype_url.size() < 2) continue;
            string logtype(logtype_url[0]), url(logtype_url[1]);
            if (startswith(url, "http://news.") || startswith(url, "http://ent.") || startswith(url, "http://photo.")) continue;
            if (startswith(url, "http://www.xxhh.com") || startswith(url, "http%3A") || startswith(url, "http://tv.sohu.com")) continue;
            if (startswith(url, "http://bbs.qianyan001.com") || startswith(url, "http://www.3jy.com") || startswith(url, "http://www.77mh.com")) continue;
            if (startswith(url, "http://novel") || startswith(url, "file:") || startswith(url, "http://ad.") || startswith(url, "http://img")) continue;

            string domain;
            int ret = getdomain(url, domain);
            if (ret == -1) continue;
            if (ifIn(domain, cheat_domain)) continue;
            if (ifIn(domain, shield_domain)) continue;
            if (*(url.end() - 1) == '/') url.erase(url.end() - 1);
            if (logtype == "bid_unbid")
                featureset.insert(logtype+""+url);

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
                featureset.insert(string(newurl));
            }
        }
        poscnt += 1;
        for (set<string>::iterator sit = featureset.begin(); sit != featureset.end(); ++sit){
            pos_it = feature2poscnt.find(*sit);
            bkg_it = feature2bkgcnt.find(*sit);
            if (pos_it == feature2poscnt.end())
                feature2poscnt[*sit] = 1;
            else
                pos_it->second += 1;
            if (bkg_it == feature2bkgcnt.end())
                feature2bkgcnt[*sit] = 1;
            else
                bkg_it->second += 1;
        }
    }
    ifs_pos_feature_log.close();
    if (poscnt == 0) return -1;

/*
    for (map<string, int>::iterator it=feature2poscnt.begin();it!= feature2poscnt.end();++it)
        cout << it->first <<"\t"<< it->second <<endl;
*/
    ifstream ifs_neg_feature_log((dirpath + "/feature_0_log").c_str());
    
    while (getline(ifs_neg_feature_log, line)){
        istringstream iss;
        iss.str(line);

        string pyid, urllist;
        iss >> pyid;
        iss >> urllist;

        set<string> featureset;
        string delim("");
        vector<string> v_url;
        v_url.reserve(100);

        split(urllist, delim, &v_url);
        for (vector<string>::iterator it = v_url.begin(); it < v_url.end(); ++it){
            vector<string> logtype_url;
            split(*it, "", &logtype_url);
            if (logtype_url.size() < 2) continue;
            string logtype(logtype_url[0]), url(logtype_url[1]);
            if (startswith(url, "http://news.") || startswith(url, "http://ent.") || startswith(url, "http://photo.")) continue;
            if (startswith(url, "http://www.xxhh.com") || startswith(url, "http%3A") || startswith(url, "http://tv.sohu.com")) continue;
            if (startswith(url, "http://bbs.qianyan001.com") || startswith(url, "http://www.3jy.com") || startswith(url, "http://www.77mh.com")) continue;
            if (startswith(url, "http://novel") || startswith(url, "http://ad.") || startswith(url, "http://img")) continue;

            string domain;
            int ret = getdomain(url, domain);
            if (ret == -1) continue;
            if (ifIn(domain, cheat_domain)) continue;
            if (ifIn(domain, shield_domain)) continue;
            if (*(url.end() - 1) == '/') url.erase(url.end() - 1);
            if (logtype == "bid_unbid")
                featureset.insert(logtype+""+url);

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
                featureset.insert(string(newurl));
            }
        }
        negcnt += 1;
        // map<string, int>::iterator bkg_it;
        for (set<string>::iterator sit = featureset.begin(); sit != featureset.end(); ++sit){
            // neg_it = feature2negcnt.find(*sit);
            bkg_it = feature2bkgcnt.find(*sit);
            //if (neg_it == feature2negcnt.end())
            //    feature2negcnt[*sit] = 1;
           // else
            //    neg_it->second += 1;
            if (bkg_it == feature2bkgcnt.end())
                feature2bkgcnt[*sit] = 1;
            else
                bkg_it->second += 1;
        }
    }
    ifs_neg_feature_log.close();
/*
    for (map<string, int>::iterator it=feature2negcnt.begin();it!= feature2negcnt.end();++it)
        cout << it->first <<"\t"<< it->second <<endl;
    double duration = total_time / CLOCKS_PER_SEC;
    cout << "split cost time:" << duration <<" seconds" << endl;
    return 0;
*/

    float ctr_thrshld = 0.2, lift_thrshld = 0.7, bkgcnt_thrshld = 3;
    if (poscnt > 6500 && poscnt < 10000)
        bkgcnt_thrshld = 4;
    else if (poscnt >= 10000)
        bkgcnt_thrshld = 5;

    int bkgcnt = poscnt + negcnt;
    // sort(feature2poscnt.begin(), feature2poscnt.end(), mysortfunc);
    for (map<string, int>::iterator it = feature2poscnt.begin(); it != feature2poscnt.end(); ++it){
        string feature = it->first;
        if (feature2poscnt[feature] == feature2bkgcnt[feature])
            continue;
        if (feature2poscnt[feature] == 0) continue;
        float ctr = (feature2poscnt[feature] + 1) * 1.0 / (feature2bkgcnt[feature] + 5);
        float posrate = feature2poscnt[feature] * 1.0 / poscnt;
        float bkgrate = feature2bkgcnt[feature] * 1.0 / bkgcnt;
        float lift = posrate / (posrate + bkgrate);

        //if (feature2bkgcnt.find(feature) == feature2bkgcnt.end())
        if (feature2bkgcnt.find(feature) == feature2bkgcnt.end() || lift < lift_thrshld)
            continue;
        if (poscnt > 5000){
            if (feature2poscnt[feature] <= 1) continue;
        }
        if (poscnt > 25000){
            if (feature2bkgcnt[feature] - feature2poscnt[feature] <= 1) continue;
        }
        if (feature2bkgcnt[feature] < bkgcnt_thrshld && feature2poscnt[feature] <= 2) continue;
        cout << feature <<" "<< feature2poscnt[feature] <<" "<< poscnt <<" "<< posrate
                        <<" "<< feature2bkgcnt[feature] <<" "<< bkgcnt <<" "<< bkgrate
                        <<" "<< lift <<" "<< ctr << endl;
    }
    return 0;
}

