#ifndef TEST_TASK_H
#define TEST_TASK_H

#include <iostream>
#include "pc_task.h"

using namespace std;

class test_task : public pc_task
{
public:
    int line_num = 0;
    test_task(){}
    virtual void run_task(vector<fm_sample>& dataBuffer)
    {
        line_num += dataBuffer.size();
        cout << dataBuffer.size() << " " << dataBuffer[0].queryId;
        for(int j = 0; j < dataBuffer.size();j++) {
            fm_sample s = dataBuffer[j];
           // cout << s.y << "===" << s.queryId << " ";
        }
        cout << endl;
        cout << ">>>>>>>>>>>>>" << line_num << endl;
    }
};


#endif //TEST_TASK_H
