#include "oneflow/core/distributed_runtime/grpc_remote_master.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"
#include "oneflow/core/distributed_runtime/grpc_master_service.h"

#include <fstream>

#include "grpc++/grpc++.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#ifdef _MSC_VER
DEFINE_string(resource,
  "D:/users/xiaoshu/sandbox/oneflow/oneflow/core/proto/cluster_config.txt", "");
#else
DEFINE_string(resource,
  "/home/xiaoshu/dl_sys/oneflow_dev_grpc/oneflow/core/proto/cluster_config.txt", "");
#endif

namespace oneflow {

TEST(GrpcMasterServer, test) {
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

  std::string server_address("127.0.0.1:50051");
  std::shared_ptr<::grpc::Channel> dst_channel
    = channel->FindChannel(server_address);

  GrpcRemoteMaster* remote_master = new GrpcRemoteMaster(dst_channel);
  oneflow::SendGraphRequest req;
  req.set_tmp(7);
  oneflow::SendGraphResponse resp;

  ::tensorflow::Status s = remote_master->SendGraph(&req, &resp);
  if (s.ok()) {
    std::cout << "1th response from server: " << resp.tmp() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  req.set_tmp(3);
  s = remote_master->SendGraph(&req, &resp);
  if (s.ok()) {
    std::cout << "2nd response from server: " << resp.tmp() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }
}  // TEST

}  // namespace oneflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


