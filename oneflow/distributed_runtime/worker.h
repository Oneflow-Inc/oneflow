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

    void GetMachineDesc(const GetMachineDescRequest* request,
                        const GetMachineDescResponse* response);

    void GetMemoryDescHandler(const GetMemoryDescRequest* request,
                              const GetMemoryDescResponse* response);
};

}
#endif /* !WORKER_H */
