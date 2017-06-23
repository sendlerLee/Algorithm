#include "pc_frame.h"
#include "test_task.h"

int main()
{
    test_task task;
    pc_frame frame;
    frame.init(task, 3, 50, 10000);
    frame.run();
    return 0;
}

