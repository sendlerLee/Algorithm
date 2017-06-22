all:
	g++ -O3 fm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/fm_train -lpthread
	#/home/xiaoju/gcc-release/gcc-release-4.8.5/bin/g++ -O3 fm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/fm_train -lpthread
	g++ -O3 fm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/fm_predict -lpthread
	#/home/xiaoju/gcc-release/gcc-release-4.8.5/bin/g++ -O3 fm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/fm_predict -lpthread
