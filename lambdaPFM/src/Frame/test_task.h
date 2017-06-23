#ifndef TEST_TASK_H
#define TEST_TASK_H

#include <iostream>
#include "pc_task.h"

using namespace std;

class test_task : public pc_task
{
public:
    test_task(){}
    virtual void run_task(vector<vector<fm_sample>>& dataBuffer)
    {
        cout <<" ==========\n";
        for(int i = 0; i < dataBuffer.size(); ++i)
        {
            cout << dataBuffer[i].size() << " " ;
            for(int j = 0; j < dataBuffer[i].size();j++) {
                fm_sample s = dataBuffer[i][j];
                cout << s.y << "===" << s.queryId << " ";
            }
            cout << endl;
        }
        cout << "**********\n";
    }
};


#endif //TEST_TASK_H
