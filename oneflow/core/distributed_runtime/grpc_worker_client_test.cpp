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

  ::grpc::CompletionQueue cq;
  GrpcRemoteWorker* remote_worker
    = new GrpcRemoteWorker(std::move(dst_channel), &cq);
  ::tensorflow::Status s;

  // Test
  oneflow::GetStatusRequest req;
  req.set_status_test("1th get status from client");
  oneflow::GetStatusResponse resp;
  s = remote_worker->GetStatus(&req, &resp);
  if (s.ok()) {
    std::cout << "1th response from server: "
      << resp.status_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  req.set_status_test("2nd get status from client");
  s = remote_worker->GetStatus(&req, &resp);
  if (s.ok()) {
    std::cout << "2nd response from server: "
      << resp.status_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  // Test for GetMachineDesc
  oneflow::GetMachineDescRequest req_machine;
  req_machine.set_machine_desc_test("get machine_desc_test from client");
  oneflow::GetMachineDescResponse resp_machine;
  s = remote_worker->GetMachineDesc(&req_machine, &resp_machine);
  if (s.ok()) {
    std::cout << resp_machine.machine_desc_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  oneflow::GetMemoryDescRequest req_memory;
  req_memory.set_memory_desc_test("get memory_desc_test from client");
  oneflow::GetMemoryDescResponse resp_memory;
  s = remote_worker->GetMemoryDesc(&req_memory, &resp_memory);
  if (s.ok()) {
    std::cout << resp_memory.memory_desc_test() << std::endl;
  } else {
    std::cout << "s is not ok" << std::endl;
  }

  oneflow::SendTaskGraphRequest req_stg;
  req_stg.set_send_task_graph_test("send_task_graph_test from client");
  oneflow::SendTaskGraphResponse resp_stg;
  s = remote_worker->SendTaskGraph(&req_stg, &resp_stg);
  if (s.ok()) {
    std::cout << resp_stg.send_task_graph_test() << std::endl;
  } else {
    std::cout << "s is not ok " << std::endl;
  }

  // Async request
  GrpcClientCQTag* callback_tag;
  void* tag;
  bool ok;
  // sendmessage request
  oneflow::SendMessageRequest req_sendmessage;
  req_sendmessage.set_send_message_test("hi~, server, I am send_message client");
  oneflow::SendMessageResponse resp_sendmessage;
  auto cb_send_message = [&resp_sendmessage](::tensorflow::Status s) {
    std::cout <<"callback : " << resp_sendmessage.send_message_test() << std::endl;
  };
  remote_worker->SendMessageAsync(&req_sendmessage, &resp_sendmessage, cb_send_message);
  remote_worker->cq_->Next(&tag, &ok);
  callback_tag  = static_cast<GrpcClientCQTag*>(tag);
  callback_tag->OnCompleted(ok);

  // readdata request
  oneflow::ReadDataRequest req_readdata;
  req_readdata.set_read_data_test("hi~, server, I am read_data client");
  oneflow::ReadDataResponse resp_readdata;
  auto cb_read_data = [&resp_readdata](::tensorflow::Status s) {
    std::cout << "callback: " << resp_readdata.read_data_test() << std::endl;
  };
  remote_worker->ReadDataAsync(&req_readdata, &resp_readdata, cb_read_data);
  remote_worker->cq_->Next(&tag, &ok);
  callback_tag = static_cast<GrpcClientCQTag*>(tag);
  callback_tag->OnCompleted(ok);
}  // TEST

}  // namespace oneflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


