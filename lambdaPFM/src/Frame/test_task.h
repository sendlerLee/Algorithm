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
    virtual void run_task(vector<vector<fm_sample>>& dataBuffer)
    {
        for(int i = 0; i < dataBuffer.size(); ++i)
        {
            line_num += dataBuffer[i].size();
            cout << dataBuffer[i][0].queryId << " " << dataBuffer[i].size() << " " << line_num << endl;
            for(int j = 0; j < dataBuffer[i].size();j++) {
                fm_sample s = dataBuffer[i][j];
            }
        }
    }
};


#endif //TEST_TASK_H
