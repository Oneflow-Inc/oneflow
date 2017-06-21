#include <fstream>
#include "oneflow/core/distributed_runtime/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"
#include "oneflow/core/distributed_runtime/grpc_worker_service.h"

#include "grpc++/grpc++.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(resource, "/home/xiaoshu/dl_sys/oneflow_dev_grpc/oneflow/core/proto/cluster_config.txt", "");

namespace oneflow {

TEST(GrpcWorkerServer, test) {
  oneflow::ClusterSpec cluster_spec;
  std::ifstream fin(FLAGS_resource);
  google::protobuf::io::IstreamInputStream pbin(&fin);
  if (!google::protobuf::TextFormat::Parse(&pbin, &cluster_spec)) {
    std::cout << "parse proto error!" << std::endl;
    fin.close();
  }
  for (auto& node_info : cluster_spec.node_info()) {
    std::cout << node_info.ip() << ":" << node_info.port() << std::endl;
  }

  GrpcChannelCache* channel = new GrpcChannelCache(cluster_spec);
  channel->CreateChannelCache();
  Worker* worker = new Worker(channel);

  std::string server_address("0.0.0.0:50051");

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  GrpcWorkerService* worker_service = new GrpcWorkerService(worker, &builder);
  std::shared_ptr<Server> server = builder.BuildAndStart();

  void* tag;
  bool ok;
  UntypedCall<GrpcWorkerService>::Tag* callback_tag = nullptr;
  /*
  // process 1th request from client
  std::cout << "server wait for 1th request......." << std::endl;
  worker_service->EnqueueGetStatusMethod();
  worker_service->cq_->Next(&tag, &ok);
  callback_tag = static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
  if (callback_tag) {
    callback_tag->OnCompleted(worker_service, ok);
  } else {
    worker_service->cq_->Shutdown();
  }
  worker_service->cq_->Next(&tag, &ok);

  // process 2nd request from client
  std::cout << "server wait for 2nd request......." << std::endl;
  worker_service->cq_->Next(&tag, &ok);
  callback_tag = static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
  if (callback_tag) {
    callback_tag->OnCompleted(worker_service, ok);
  } else {
    worker_service->cq_->Shutdown();
  }
  worker_service->cq_->Next(&tag, &ok);

  //process GetMachineDesc from client
  worker_service->EnqueueGetMachineDescMethod();
  worker_service->cq_->Next(&tag, &ok);
  callback_tag = static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
  if (callback_tag) {
    callback_tag->OnCompleted(worker_service, ok);
  } else {
    worker_service->cq_->Shutdown();
  }
  worker_service->cq_->Next(&tag, &ok);
  
  //process GetMemoryDesc from client
  worker_service->EnqueueGetMemoryDescMethod();
  worker_service->cq_->Next(&tag, &ok);
  callback_tag = static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
  if (callback_tag) {
    callback_tag->OnCompleted(worker_service, ok);
  } else {
    worker_service->cq_->Shutdown();
  }
  worker_service->cq_->Next(&tag, &ok);
  */ 
  // process SendTaskGraph from client
  worker_service->EnqueueSendTaskGraphMethod();
  worker_service->cq_->Next(&tag, &ok);
  callback_tag = static_cast<UntypedCall<GrpcWorkerService>::Tag*>(tag);
  if (callback_tag) {
    callback_tag->OnCompleted(worker_service, ok);
  } else {
    worker_service->cq_->Shutdown();
  }
  worker_service->cq_->Next(&tag, &ok);

  delete worker_service;
}  // TEST

}  // namespace oneflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
