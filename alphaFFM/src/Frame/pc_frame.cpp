#include "pc_frame.h"

bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    pTask = &task;
    threadNum = t_num;
    bufSize = buf_size;
    logNum = log_num;
    finished_flag = false;
    sem_init(&semPro, 0, 0);
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
    while(true)
    {
        while(getline(cin,line))
        {
            line_num++;
            buffer.push(line);
            sem_post(&semCon);
            if(line_num%logNum == 0)
            {
                cout << line_num << " lines have finished" << "\r";
            }
            sem_wait(&semPro);
        }
        finished_flag = true;
        sem_post(&semCon);
        if(finished_flag) {
            break;
        }
    }
    cout <<"total " << line_num << " query samples have finished" << endl;
}


void pc_frame::conThread(){
    vector<string> input_vec;
    input_vec.reserve(bufSize);
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
        bufMtx.unlock();
        sem_post(&semPro);
        if(input_vec.size() > 0) {
            pTask->run_task(input_vec);
        }
        if(finished_flag)
            break;
    }
    sem_post(&semCon);
}

