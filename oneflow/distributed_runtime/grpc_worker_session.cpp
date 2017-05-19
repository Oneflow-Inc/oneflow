/*
 * grpc_worker_session.cpp
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "distributed_runtime/grpc_worker_session.h"
#include "distributed_runtime/grpc_remote_worker.h"

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
  

}

void GrpcWorkerSession::GetMemoryDesc() {

}

}



