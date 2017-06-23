#include "pc_frame.h"
#include "../Sample/fm_sample.h"

bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    pTask = &task;
    threadNum = t_num;
    bufSize = buf_size;
    logNum = log_num;
    finished_flag = false;
    sem_init(&semPro, 0, 1);
    sem_init(&semCon, 0, 0);
    threadVec.clear();
    threadVec.push_back(thread(&pc_frame::proThread, this));
    for(int i = 0; i < threadNum; ++i)
    {
        threadVec.push_back(thread(&pc_frame::conThread, this));
    }
    return true;
}


void pc_frame::run()
{
    for(int i = 0; i < threadVec.size(); ++i)
    {
        threadVec[i].join();
    }
}

void pc_frame::proThread()
{
    string line;
    int line_num = 0;
    int i = 0;
    string lastID = "";
    vector<fm_sample> samples;
    string className = typeid (*pTask).name();
    size_t position = className.find("predictor");
    //cout << "class name is : " << className << " " << (position != string::npos) << endl;
    while(true)
    {
        sem_wait(&semPro);
        bufMtx.lock();
        bool hasRef = false;
        while(true)
        {
            if(!getline(cin, line))
            {
                finished_flag = true;
                break;
            }
            fm_sample sample(line);
            line_num ++; 
            //cout << lastID << "====== " << sample.queryId << "======== " << sample.y << endl;
            if(!lastID.empty() && lastID != sample.queryId){
                if(hasRef || (position == string::npos)) {
                    buffer.push(samples);
                    if(line_num%logNum == 0)
                    {
                        cout << line_num << " samples have finished" << endl;
                    }
                }
                bufMtx.unlock();
                sem_post(&semCon);
                samples.clear();
                hasRef = false;
            }
            lastID = sample.queryId;
            if(sample.y > 0) {
                hasRef = true;
            }
            samples.push_back(sample);
        }
        if(finished_flag)
        {
            buffer.push(samples);
            bufMtx.unlock();
            sem_post(&semCon);
            cout <<"total " << line_num << " samples have finished" << endl;
            break;
        }
    }
}


void pc_frame::conThread(){
    vector<vector<fm_sample> > input_vec;
    input_vec.reserve(buffer.size());
    int line_nu = 0;
    while(true)
    {
        input_vec.clear();
        sem_wait(&semCon);
        bufMtx.lock();
        for(int i = 0; i < buffer.size(); ++i)
        {
            input_vec.push_back(buffer.front());
            buffer.pop();
        }
        line_nu += input_vec[0].size();
        bufMtx.unlock();
        sem_post(&semPro);
        if(input_vec.size() > 0) {
            pTask->run_task(input_vec);
        }
        if(finished_flag) {
            cout << line_nu << "  samples ." << endl;
            break;
        }
    }
    sem_post(&semCon);
}

