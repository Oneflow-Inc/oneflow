/*
 * worker.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef WORKER_H
#define WORKER_H

#include "distributed_runtime/worker_service.pb.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

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

    template <typename ProtoMessage>
    void ParseToProto(ProtoMessage& proto_type, std::string& file_name);

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

    std::string machine_desc_file_path;
    std::string memory_desc_file_path; 

};


}
#endif /* !WORKER_H */
