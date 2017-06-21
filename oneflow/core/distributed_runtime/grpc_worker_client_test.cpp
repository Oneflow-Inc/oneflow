#include "oneflow/core/distributed_runtime/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/worker_service.pb.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"
#include "oneflow/core/distributed_runtime/grpc_worker_service.h"

#include <fstream>

#include "grpc++/grpc++.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

DEFINE_string(resource,
    "/home/xiaoshu/dl_sys/oneflow_dev_grpc/oneflow/core/proto/cluster_config.txt", "");

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

  std::string server_address("0.0.0.0:50051");
  std::shared_ptr<::grpc::Channel> dst_channel
    = channel->FindChannel(server_address);

  std::unique_ptr< ::grpc::ServerCompletionQueue> cq_;
  GrpcRemoteWorker* remote_worker = new GrpcRemoteWorker(dst_channel, cq_.get());
  ::tensorflow::Status s;

  //Test
  oneflow::GetStatusRequest req;
  oneflow::GetStatusResponse resp;
  s = remote_worker->GetStatus(&req, &resp);
  if (s.ok()) {
    std::cout << "1th response from server: " << resp.status_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  s = remote_worker->GetStatus(&req, &resp);
  if (s.ok()) {
    std::cout << "2nd response from server: " << resp.status_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  //Test for GetMachineDesc
  oneflow::GetMachineDescRequest req_machine;
  oneflow::GetMachineDescResponse resp_machine;
  s = remote_worker->GetMachineDesc(&req_machine, &resp_machine);
  if (s.ok()) {
    std::cout<<resp_machine.machine_desc_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  oneflow::GetMemoryDescRequest req_memory;
  oneflow::GetMemoryDescResponse resp_memory;
  s = remote_worker->GetMemoryDesc(&req_memory, &resp_memory);
  if (s.ok()) {
    std::cout<<resp_memory.memory_desc_test() << std::endl;
  } else {
    std::cout<< "s is not ok" << std::endl;
  }

  oneflow::SendTaskGraphRequest req_stg;
  //req_sendtask.set_send_task_graph_test("send_task_graph_test from client");
  oneflow::SendTaskGraphResponse resp_stg;
  s = remote_worker->SendTaskGraph(&req_stg, &resp_stg);
  if (s.ok()) {
    std::cout<< resp_stg.send_task_graph_test() << std::endl;
  } else {
    std::cout << "s is not ok " << std::endl;
  }

  /*
  oneflow::SendMessageRequest req_sendmessage;
  oneflow::SendMessageResponse resp_sendmessage;
  s = remote_worker->SendMessage(&req_sendmessage, &resp_sendmessage);
  if (s.ok()) {
    std::cout<<resp_sendmessage.send_message_test() << std::endl;
  } else {
    std::cout << "s is not ok " << std::endl;
  }
  */

}  // TEST

}  // namespace oneflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


