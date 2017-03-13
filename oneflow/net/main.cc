#include <iostream>

#include "server_lib.h"
#include "oneflow_server.pb.h"

using oneflow::ClusterDef;
using oneflow::JobDef;
using oneflow::ServerDef;

int main(){
  ClusterDef cluster;
  JobDef job;
  ServerDef server;

  job.set_name("ps");
  auto task = job.mutable_tasks();
  std::string tmp_task = "worker1.example.com:2222";
  (*task)[1] = tmp_task;
  
  return 0;
}
