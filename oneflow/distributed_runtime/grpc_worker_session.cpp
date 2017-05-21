/*
 * grpc_worker_session.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_worker_session.h"

namespace oneflow {

GrpcWorkerSession::GrpcWorkerSession() {}
GrpcWorkerSession::~GrpcWorkerSession() {}

void GrpcWorkerSession::GetMachineDesc() {
  oneflow::MachineDesc machine_desc;
  machine_desc.set_machine_id(0);
  machine_desc.set_ip("192.168.1.12");
  machine_desc.set_port(50051);

  oneflow::GetMachineDescRequest req;
  req.set_allocated_machine_desc(&machine_desc);

  oneflow::GetMachineDescResponse resp;
  auto cb = [] () {
    //TODO
    //get info from response that back from server
  };
  remote_worker->GetMachineDescAsync(&req, &resp, cb);
  

}

void GrpcWorkerSession::GetMemoryDesc() {
  oneflow::MemoryDesc memory_desc;
  memory_desc.set_machine_id(0);
  memory_desc.set_memory_address(555);
  memory_desc.set_remoted_token(666);
  oneflow::GetMemoryDescRequest req;
  req.set_allocated_memory_desc(&memory_desc);
  
  oneflow::GetMemoryDescResponse resp;
  auto cb = [resp] () {
    oneflow::MemoryDesc memory_desc_resp;
    memory_desc_resp.set_machine_id(resp.memory_desc().machine_id());
    memory_desc_resp.set_memory_address(resp.memory_desc().memory_address());
    memory_desc_resp.set_remoted_token(resp.memory_desc().remoted_token());
    //get info from response that back from server
  };
  remote_worker->GetMemoryDescAsync(&req, &resp, cb); 
}

}



