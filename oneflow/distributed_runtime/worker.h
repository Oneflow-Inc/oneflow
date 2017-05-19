/*
 * worker.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef WORKER_H
#define WORKER_H

#include "distributed_runtime/worker_service.pb.h"

namespace oneflow {

class Worker {
  public:
    Worker();
    ~Worker() {};

    void GetMachineDesc(GetMachineDescRequest* request,
                        GetMachineDescResponse* response);

    void GetMemoryDesc(GetMemoryDescRequest* request,
                              GetMemoryDescResponse* response);

    void SendMessage(SendMessageRequest* request,
                     SendMessageResponse* response);
    void ReadData(ReadDataRequest* request,
                  ReadDataResponse* response);
};


}
#endif /* !WORKER_H */
