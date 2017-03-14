#include <iostream>
#include "server_lib.h"

using tensorflow::ClusterDef;
using tensorflow::JobDef;
using tensorflow::ServerDef;

int main(){
  ServerDef server;

  ClusterDef* cluster = server.mutable_cluster();
  server.set_job_name("oneflow_test");
  server.set_task_index(0);
  server.set_protocol("grpc"); 
  
  JobDef* job = cluster->add_job();
  job->set_name("ps");
  auto task = job->mutable_tasks();
  int key = 0; 
  std::string val = "ps0.example:2222";
  (*task)[key] = val;
 
  job = cluster->add_job(); 
  job->set_name("ps");
  task = job->mutable_tasks();
  key = 1;
  val = "ps1.example:2222";
  (*task)[key] = val;

  job = cluster->add_job();
  job->set_name("worker");
  task = job->mutable_tasks();
  key = 0;
  val = "worker0.example:2223";
  (*task)[key] = val; 

  job = cluster->add_job();
  job->set_name("worker");
  task = job->mutable_tasks();
  key = 1;
  val = "worker1.example:2223";
  (*task)[key] = val;

  job = cluster->add_job();
  job->set_name("worker");
  task = job->mutable_tasks();
  key = 2;
  val = "worker2.example:2223";
  (*task)[key] = val;  
  
  int length = server.ByteSize();
  char* buf = new char[length];
  server.SerializeToArray(buf, length);

  ServerDef serverparse;
  serverparse.ParseFromArray(buf, length);
  std::cout<<serverparse.job_name()<<std::endl;
  std::cout<<serverparse.task_index()<<std::endl;
 
  auto pcluster = serverparse.cluster();
  auto pjob = pcluster.mutable_job();
  for(auto iter = pjob->begin(); iter != pjob->end(); ++iter){
    std::cout<<iter->name()<<" ";
    for(auto task : iter->tasks()){
      std::cout<<task.first<<" "<<task.second<<std::endl;
    }
  }
  return 0;
}
