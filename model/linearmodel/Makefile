CXX = g++
CXXFLAGS = -Wall -Wconversion -O2 -fPIC -std=c++0x -march=native -fopenmp
MAIN = gbdt
FILES = common.cpp timer.cpp gbdt.cpp
SRCS = $(FILES:%.cpp=%.cpp)
HEADERS = $(FILES:%.cpp=%.h)

all: $(MAIN)

gbdt: train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)

g++ -O3 -std=c++0x -o  LogisticRegression LogisticRegression.cpp 
g++ -O3 -std=c++0x -fopenmp -o  PLR PLR.cpp 
g++ -O3 -std=c++0x -fopenmp -o  SSFALR2 SSFALR2.cpp 
g++ -O3 -std=c++0x -fopenmp -o  PLogitBoost PLogitBoost.cpp 
g++ -O3 -std=c++0x -fopenmp -o  FTRLProximalBatch FTRLProximalBatch.cpp 
g++ -O3 -std=c++0x -fopenmp -o  FTRLProximal FTRLProximal.cpp 
