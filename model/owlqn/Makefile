CXX = g++
CXXFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++0x -march=native -fopenmp
MAIN = owlqn
FILES = leastSquares.cpp logreg.cpp OWLQN.cpp TerminationCriterion.cpp 
SRCS = $(FILES:%.cpp=%.cpp)
HEADERS = $(FILES:%.cpp=%.h)

all: $(MAIN)

owlqn: main.cpp $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SRCS)

clean:
	rm -f $(MAIN)
