#include "oneflow/core/distributed_runtime/master.h"

namespace oneflow {

Master::Master(GrpcChannelCache* channel_cache) 
  : channel_cache_(channel_cache) {}

Master::~Master() {}

::tensorflow::Status Master::SendGraph(SendGraphRequest* request,
                       SendGraphResponse* response) {
  /*
  GraphCompile graph_compile;
  graph_compile->compile();
  */
  //oneflow::SendTaskGraphRequest task_request;
  //oneflow::SendTaskGraphResponse task_response;
  //Barrier();
  std::cout<<"Server: request from Client = "<<request->tmp()<<std::endl;
  response->set_tmp(8);
  //for(auto& channel : channel_cache_->channel_map_) {
  //  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
    //remote_worker_ = new GrpcRemoteWorker(channel.second, cq_.get());
    //remote_worker_->SendTaskGraphSync(task_request, task_response);
  //}
  //std::cout<<"master: status = "<<::tensorflow::Status::OK()<<std::endl;
  return ::tensorflow::Status::OK();
  //return true;
}

void Master::Barrier() {
  //oneflow::CheckStatusRequest request;
  //oneflow::CheckStatusResponse response;
  int32_t worker_num = 0;
  for(auto& channel : channel_cache_->channel_map_) {
    std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
    //remote_worker_ = new GrpcRemoteWorker(channel.second, cq_.get());
    //remote_worker_->CheckStatusSync(request, response);
    //worker_num += response.status();
  }
  if(worker_num != channel_cache_->channel_map_.size()) return;
}

}



