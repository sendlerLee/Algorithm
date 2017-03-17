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
