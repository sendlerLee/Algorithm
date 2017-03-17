CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native -fopenmp
MAIN = randomforest
FILES = common.cpp timer.cpp randomforest.cpp
SRCS = $(FILES:%.cpp=%.cpp)
HEADERS = $(FILES:%.cpp=%.h)

all: $(MAIN)

randomforest: train.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
