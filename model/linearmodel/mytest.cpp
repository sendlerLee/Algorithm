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

//author: tannaiqiang (532429163@qq.com)

using namespace std;

void loadTrainInstance() 
{
	string filename = "campaignctrmodel_train_20151016_2_days_ns1_100w";
	//string filename = "campaignctrmodel_test_20151017_ufs1_100w.libfm";
	std::ifstream fData(filename.c_str());
	if (! fData.is_open()) 
	{
		throw "unable to open " + filename;
	}

	int nchar;
	while (!fData.eof()) 
	{
		std::string line;
		std::getline(fData, line);
		const char *pline = line.c_str();
		cout << pline << endl;

		char username[1000];
		int y = 0;
		if (sscanf(pline, "%s%d%n", &username, &y, &nchar) >= 1) 
		{
			pline += nchar;
			cout << "username: " << username << endl;
			cout << "y: " << y << endl;
			
			char feat[1000];
			while (sscanf(pline, "%s%n", &feat, &nchar) >= 0) 
			{
				string feature = feat;
				int findex = feature.find_last_of(":");

				string featname = feature.substr(0, findex).c_str();
				double x = atof(feature.substr(findex + 1, feature.size() - findex).c_str());

				pline += nchar;	
				cout << "fdsa\t" << featname << "\t" << x << endl;
			}
		}
	}
	fData.close();
}

int main(int const argc, char const * const * const argv) 
{
	loadTrainInstance();

	return 0;
}
