#include <fstream>
#include "oneflow/core/distributed_runtime/grpc_remote_master.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"
#include "oneflow/core/distributed_runtime/grpc_master_service.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(resource, "/home/xiaoshu/dl_sys/oneflow/oneflow/proto/resource.txt", "");

namespace oneflow {

TEST(GrpcMasterServer, test) {
  oneflow::ClusterSpec cluster_spec;
  std::ifstream fin(FLAGS_resource);
  google::protobuf::io::IstreamInputStream pbin(&fin);
  if(!google::protobuf::TextFormat::Parse(&pbin, &cluster_spec)) {
    fin.close();
  }

  GrpcChannelCache* channel = new GrpcChannelCache(cluster_spec);

  Master* master = new Master(channel);
  std::string server_address("0.0.0.0:500051");
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  GrpcMasterService* master_service = new GrpcMasterService(master, &builder);
  builder.BuildAndStart();

  master_service->EnqueueSendGraphMethod();

}

}

