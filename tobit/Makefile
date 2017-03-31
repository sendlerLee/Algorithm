CXX = g++
CXXFLAGS = -Wall -Wconversion -O2 -fPIC -std=c++11 -march=native -fopenmp
MAIN = tobit
SRCS = lbfgs.c
HEADERS = lbfgs.h arithmetic_ansi.h arithmetic_sse_double.h arithmetic_sse_float.h

all: $(MAIN)

tobit: tobit.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
