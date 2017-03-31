#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;

static double pnorm(double x)
{
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;

        int sign = 1;
        if (x < 0)
            sign = -1;
        x = fabs(x)/sqrt(2.0);

        double t = 1.0/(1.0 + p*x);
        double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

        return 0.5*(1.0 + sign*y);
}

int main(int argc, char* argv[]) {
	double i = 0,j = 0;
	cout << i/j << "\t" << i/j * 0 <<  endl;
	for(int i = 0; i < argc; i++) {
		cout << pnorm(atof(argv[i])) << " " << log(pnorm(atof(argv[i])))<< endl;	
	}
}
