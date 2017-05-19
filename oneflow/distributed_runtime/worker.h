/*
 * worker.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef WORKER_H
#define WORKER_H

#include "distributed_runtime/worker_service.pb.h"
#include "network/grpc/grpc_worker.h"

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

  private:
    struct machine_desc {
      int32_t machine_id;
      std::string ip;
      int32_t port;
    };
    machine_desc machine_desc_;

    struct memory_desc {
      int64_t machine_id;
      int64_t memory_address;
      int64_t remote_token;
    };
    memory_desc memory_desc_;

};


}
#endif /* !WORKER_H */
