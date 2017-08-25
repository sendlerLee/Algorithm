all:
	/home/xiaoju/gcc-release/gcc-release-4.8.5/bin/g++ -O3 fm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -DUSEOMP -fopenmp -o bin/ffm_train -lpthread
	/home/xiaoju/gcc-release/gcc-release-4.8.5/bin/g++ -O3 fm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -DUSEOMP -fopenmp -o bin/ffm_predict -lpthread
