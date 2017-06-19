#include <fstream>
#include "oneflow/core/distributed_runtime/grpc_remote_master.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"
#include "oneflow/core/distributed_runtime/grpc_master_service.h"

#include "grpc++/grpc++.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(resource, "/home/xiaoshu/dl_sys/oneflow_dev_grpc/oneflow/core/proto/cluster_config.txt", "");

namespace oneflow {

TEST(GrpcMasterServer, test) {
  oneflow::ClusterSpec cluster_spec;
  std::ifstream fin(FLAGS_resource);
  google::protobuf::io::IstreamInputStream pbin(&fin);
  if(!google::protobuf::TextFormat::Parse(&pbin, &cluster_spec)) {
    std::cout<<"parse proto error!"<<std::endl;
    fin.close();
  }
  for(auto& node_info : cluster_spec.node_info()) {
    std::cout<<node_info.ip()<<":"<<node_info.port()<<std::endl;
  }
  GrpcChannelCache* channel = new GrpcChannelCache(cluster_spec);
  channel->CreateChannelCache();
  Master* master = new Master(channel);

  std::string server_address("0.0.0.0:50051");

  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  GrpcMasterService* master_service = new GrpcMasterService(master, &builder);
  std::shared_ptr<Server> server_ = builder.BuildAndStart();

  master_service->EnqueueSendGraphMethod();
  //master_service->test();
  void* tag;
  bool ok;
  //master_service->cq_->Next(&tag, &ok);

}

}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
