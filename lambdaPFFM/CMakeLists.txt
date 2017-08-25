cmake_minimum_required(VERSION 3.7)
project(lambdaPFM)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/Frame/pc_frame.cpp
        src/Frame/pc_frame.h
        src/Frame/pc_task.h
        src/Frame/test_main.cpp
        src/Frame/test_task.h
        src/FTRL/ftrl_model.h
        src/FTRL/ftrl_predictor.h
        src/FTRL/ftrl_trainer.h
        src/Sample/fm_sample.h
        src/Utils/utils.cpp
        src/Utils/utils.h
        fm_predict.cpp
        fm_train.cpp
        Makefile
        README.md)

add_executable(lambdaPFM ${SOURCE_FILES})
