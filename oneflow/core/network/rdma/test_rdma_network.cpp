#include "gflags/gflags.h"
#include "glog/logging.h"
#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_memory.h"
#include "oneflow/core/network/network_message.h"

#include <time.h>
#include <iostream>
#ifdef WIN32
#include <Windows.h>
#endif

using namespace oneflow;
using namespace std;

DEFINE_bool(is_client, true, "whether i am the client");
DEFINE_string(my_ip, "11.11.1.109", "my machine ip");
DEFINE_string(peer_ip, "11.11.1.132", "peer machine ip");
DEFINE_int64(transfer_size, 1024, "transfer data size");
DEFINE_int32(transfer_times, 1, "transfer data times");
DEFINE_int32(port, 5551, "default port");

namespace oneflow {

class RDMATest final {
 public:
  RDMATest(int64_t my_machine_id, int64_t peer_machine_id,
           const NetworkTopology& topology);
  ~RDMATest();

  int64_t round_trip() const { return round_trip_; }

  void SendMemoryDescriptor();
  bool Poll(NetworkResult* result);

  void ProcessNetworkResult(const NetworkResult& result);
  void ProcessSendOk(const NetworkResult& result);
  void ProcessReceiveMsg(const NetworkResult& result);
  void ProcessReadOk(const NetworkResult& result);

 private:
  int64_t my_machine_id_;
  int64_t peer_machine_id_;
  Network* net_;
  char* buffer_;
  NetworkMemory* network_buffer_;
  int64_t round_trip_;
};

RDMATest::RDMATest(int64_t my_machine_id, int64_t peer_machine_id,
                   const NetworkTopology& topology)
    : my_machine_id_(my_machine_id),
      peer_machine_id_(peer_machine_id),
      round_trip_(0) {
  LOG(INFO) << "Network Starting Up..." << endl;
  net_ = oneflow::GetRdmaInstance();
  LOG(INFO) << "Create Rdma Instance Success." << endl;
  net_->Init(my_machine_id_, topology);
  LOG(INFO) << "Net Init Success." << endl;

  buffer_ = new char[FLAGS_transfer_size];
  network_buffer_ = net_->RegisterMemory(buffer_, FLAGS_transfer_size);
}

RDMATest::~RDMATest() {
  delete[] buffer_;
  LOG(INFO) << "Network Shutting Down..." << endl;
}

void RDMATest::SendMemoryDescriptor() {
  NetworkMessage memory_msg;
  memory_msg.type = NetworkMessageType::kRemoteMemoryDescriptor;
  memory_msg.src_machine_id = my_machine_id_;
  memory_msg.dst_machine_id = peer_machine_id_;
  memory_msg.address = network_buffer_->memory_discriptor().address;
  memory_msg.token = network_buffer_->memory_discriptor().remote_token;
  net_->SendMsg(memory_msg);
}

bool RDMATest::Poll(NetworkResult* result) { return net_->Poll(result); }

void RDMATest::ProcessNetworkResult(const NetworkResult& result) {
  switch (result.type) {
    case NetworkResultType::kSendOk: ProcessSendOk(result); break;
    case NetworkResultType::kReceiveMsg: ProcessReceiveMsg(result); break;
    case NetworkResultType::kReadOk: ProcessReadOk(result); break;
  }
}

void RDMATest::ProcessSendOk(const NetworkResult& result) {
  LOG(INFO) << "Send network msg ok";
}

void RDMATest::ProcessReceiveMsg(const NetworkResult& result) {
  if (result.net_msg.type == NetworkMessageType::kRemoteMemoryDescriptor) {
    LOG(INFO) << "recv memory descriptor";
    MemoryDescriptor remote_memory_descriptor;
    remote_memory_descriptor.machine_id = result.net_msg.src_machine_id;
    remote_memory_descriptor.address = result.net_msg.address;
    remote_memory_descriptor.remote_token = result.net_msg.token;
    LOG(INFO) << "remote_machine_id: " << remote_memory_descriptor.machine_id;
    LOG(INFO) << "remote_address:    " << remote_memory_descriptor.address;
    LOG(INFO) << "remote_token:      " << remote_memory_descriptor.remote_token;

    net_->Read(remote_memory_descriptor, network_buffer_, []() {});
    LOG(INFO) << "async read issued";
  } else if (result.net_msg.type == NetworkMessageType::kRequestAck) {
    LOG(INFO) << "Send next memory descriptor";
    SendMemoryDescriptor();

    // Increase the number of Write success
    ++round_trip_;
  }
}

void RDMATest::ProcessReadOk(const NetworkResult& result) {
  LOG(INFO) << "Read OK";
  NetworkMessage read_ok_msg;
  read_ok_msg.type = NetworkMessageType::kRequestAck;
  read_ok_msg.src_machine_id = my_machine_id_;
  read_ok_msg.dst_machine_id = peer_machine_id_;
  net_->SendMsg(read_ok_msg);

  // Increase the number of Read success
  ++round_trip_;
}

}  // namespace oneflow

int main(int argc, char** argv) {
  google::InitGoogleLogging((const char*)argv[0]);
  google::SetLogDestination(google::GLOG_INFO, "./rdma_info");
  gflags::SetUsageMessage("Usage: ./rdma_network_test");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  int64_t client_id = 0;
  int64_t server_id = 1;
  int64_t my_machine_id = FLAGS_is_client ? client_id : server_id;
  int64_t peer_machine_id = FLAGS_is_client ? server_id : client_id;
  std::string client_ip = FLAGS_is_client ? FLAGS_my_ip : FLAGS_peer_ip;
  std::string server_ip = FLAGS_is_client ? FLAGS_peer_ip : FLAGS_my_ip;

  struct NetworkTopology net_topo;
  net_topo.all_nodes.resize(2);
  net_topo.all_nodes[client_id].machine_id = client_id;
  net_topo.all_nodes[client_id].address = client_ip;
  net_topo.all_nodes[client_id].port = FLAGS_port;
  net_topo.all_nodes[client_id].neighbors.insert(server_id);
  net_topo.all_nodes[server_id].machine_id = server_id;
  net_topo.all_nodes[server_id].address = server_ip;
  net_topo.all_nodes[server_id].port = FLAGS_port;
  net_topo.all_nodes[server_id].neighbors.insert(client_id);

  oneflow::RDMATest rdma_test(my_machine_id, peer_machine_id, net_topo);
  NetworkResult result;

  if (FLAGS_is_client) {
    for (;;) {
      if (rdma_test.Poll(&result)) {
        rdma_test.ProcessNetworkResult(result);
        if (rdma_test.round_trip() >= FLAGS_transfer_times) break;
      }
    }
  } else {
    rdma_test.SendMemoryDescriptor();
    for (;;) {
      if (rdma_test.Poll(&result)) {
        rdma_test.ProcessNetworkResult(result);
        if (rdma_test.round_trip() >= FLAGS_transfer_times) break;
      }
    }
  }

  if (FLAGS_is_client) {
    // The client has a last msg from the server
    for (;;) {
      if (rdma_test.Poll(&result)) break;
    }
  }

  gflags::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  return 0;
}
