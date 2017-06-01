#include "gflags/gflags.h"
#include "glog/logging.h"
#include "network/network.h"
#include "network/network_message.h"
#include "network/network_memory.h"

#include <iostream>

#define BUFFER_SIZE 50

using namespace oneflow;
using namespace std;

DEFINE_int32(my_machine_id, 0, "local machine id");
DEFINE_int32(peer_machine_id, 1, "peer machine id");
DEFINE_int32(transfer_size, 1024, "transfer data size");
DEFINE_int32(transfer_times, 1, "transfer data times");

int main(int argc, char** argv) {
  google::InitGoogleLogging((const char *)argv[0]);
  
  gflags::SetUsageMessage("Usage: ./rdma_network_test");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cout << "Network Starting Up..." << endl;
  Network* net = oneflow::GetRdmaInstance();
  cout << "Create Rdma Instance Success." << endl;

  struct NetworkTopology net_topo;
  net_topo.all_nodes.resize(2);
  net_topo.all_nodes[0].machine_id = 0;
  net_topo.all_nodes[0].address = "11.11.1.11";
  net_topo.all_nodes[0].port = 53433;
  net_topo.all_nodes[0].neighbors.insert(1);
  net_topo.all_nodes[1].machine_id = 1;
  net_topo.all_nodes[1].address = "11.11.1.13";
  net_topo.all_nodes[1].port = 53433;
  net_topo.all_nodes[1].neighbors.insert(0);

  // modify here manually
  uint64_t my_machine_id = FLAGS_my_machine_id;
  uint64_t peer_machine_id = FLAGS_peer_machine_id;

  net->Init(my_machine_id, net_topo);
  cout << "Net Init Success." << endl;

  NetworkMessage msg;
  NetworkResult result;
  
  /*  
  msg.src_machine_id = my_machine_id;
  msg.type = NetworkMessageType::MSG_TYPE_BARRIER;
  msg.dst_machine_id = peer_machine_id;

  net->Send(msg);
  cout << "PostSendRequest" << endl;

  int k = 0;
  for (int i = 0; i < 2 * net_topo.all_nodes[my_machine_id].neighbors.size(); ++i) {
    while (!net->Poll(&result)) {
      sleep(1);
      printf("Poll time: %d, false\n", k++);
    }
    printf("Poll time: %d, true \n", k++);
    if (result.type == NetworkResultType::NET_SEND_OK) {
      printf("Send to %d OK\n", i);
    }
    else if (result.type == NetworkResultType::NET_RECEIVE_MSG) {
      printf("Receive from %d OK\n", result.net_msg.src_machine_id);
    }
    else {
      printf("Unexpected net event polled\n");
    }
  }

  cout << "Send/Recv test success." << endl;
  */

  // useful for my_machine_id == 0
  NetworkMemory* dst_memory = net->NewNetworkMemory();
  char dst_buffer[FLAGS_transfer_size];
  dst_memory->Reset(dst_buffer, BUFFER_SIZE, my_machine_id);
  dst_memory->Register();
  MemoryDescriptor* remote_memory_descriptor = new MemoryDescriptor();
  remote_memory_descriptor->address = 0;

  // useful for my_machine_id == 1
  NetworkMemory* src_memory = net->NewNetworkMemory();
  char src_buffer[FLAGS_transfer_size];
  src_memory->Reset(src_buffer, BUFFER_SIZE, my_machine_id);
  src_memory->Register();
  // send memory descriptor to peer
  if (my_machine_id == 1) {
    NetworkMessage memory_msg;
    memory_msg.type = NetworkMessageType::MSG_TYPE_REMOTE_MEMORY_DESCRIPTOR;
    memory_msg.src_machine_id = my_machine_id;
    memory_msg.dst_machine_id = peer_machine_id;
    memory_msg.address = src_memory->memory_discriptor().address;
    memory_msg.token = src_memory->memory_discriptor().remote_token;
    net->Send(memory_msg);
  }
  
  // useful for all machine
  for (int i = 0; i < 5; ++i) {
    while (!net->Poll(&result)) {
      // sleep(1);
      // cout << "Poll result false" << endl;
    }
    if (result.type == NetworkResultType::NET_SEND_OK) {
      printf("send ok\n");
    }
    else if (result.type == NetworkResultType::NET_RECEIVE_MSG) {
      if (result.net_msg.type == NetworkMessageType::MSG_TYPE_REMOTE_MEMORY_DESCRIPTOR) {
        printf("recv descriptor\n");
        remote_memory_descriptor->machine_id = result.net_msg.src_machine_id;
        remote_memory_descriptor->address = result.net_msg.address;
        remote_memory_descriptor->remote_token = result.net_msg.token;
        if (remote_memory_descriptor->address == 0) { printf("address error\n"); exit(1); }
        net->Read(remote_memory_descriptor, dst_memory);
        printf("async read issued\n");
      }
    }
    else if (result.type == NetworkResultType::NET_READ_OK) {
      cout << "READ OK, Get memory size: " << sizeof(dst_buffer) << endl;
    }
  }

  cout << "Network Shutting Down..." << endl;
  int a;
  cin >> a;
  gflags::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  return 0;
}
