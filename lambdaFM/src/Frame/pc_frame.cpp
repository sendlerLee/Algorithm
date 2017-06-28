#include "pc_frame.h"
#include "../Sample/fm_sample.h"

bool pc_frame::init(pc_task& task, int t_num, int buf_size, int log_num)
{
    pTask = &task;
    threadNum = t_num;
    bufSize = buf_size;
    logNum = log_num;
    return true;
}


void pc_frame::run()
{
    string line;
    int line_num = 0;
    string lastID = "";
    vector<fm_sample> samples;
    string className = typeid (*pTask).name();
    size_t position = className.find("predictor");
    //cout << "class name is : " << className << " " << (position != string::npos) << endl;
    bool hasRef = false;
    while(getline(cin, line))
    {
        //cout << line << endl;
        fm_sample sample(line);
        //cout << sample.y << " " << sample.queryId << endl;
        if(!lastID.empty() && lastID != sample.queryId)
        {
            if(hasRef || (position != string::npos)) 
            {
                line_num ++; 
                pTask->run_task(samples);
                if(line_num%logNum == 0)
                {
                    cout << line_num << " query samples have finished." << "\r";
                }
            }
            samples.clear();
            hasRef = false;
        }
        lastID = sample.queryId;
        if(sample.y > 0) {
            hasRef = true;
        }
        samples.push_back(sample);
    }
    if(hasRef || (position != string::npos))
    {
        line_num ++; 
        pTask->run_task(samples);
    }
    cout <<"total " << line_num << " query samples have finished." << endl;
}

void pc_frame::conThread(){
    cout << "conThread is running." << endl;
}

void pc_frame::proThread(){
    cout << "proTHrad is runing." << endl;
}
