#include "gflags/gflags.h"
#include "glog/logging.h"
#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_message.h"
#include "oneflow/core/network/network_memory.h"

#include <time.h>
#include <iostream>
#ifdef WIN32
#include <Windows.h>
#endif

#define BUFFER_SIZE 50

using namespace oneflow;
using namespace std;

DEFINE_int32(my_machine_id, 0, "local machine id");
DEFINE_string(my_ip, "11.11.1.109", "local machine ip");
DEFINE_int32(peer_machine_id, 1, "peer machine id");
DEFINE_string(peer_ip, "11.11.1.132", "peer machine ip");
DEFINE_int64(transfer_size, 1024, "transfer data size");
DEFINE_int32(transfer_times, 1, "transfer data times");

int main(int argc, char** argv) {
  google::InitGoogleLogging((const char *)argv[0]);
  google::SetLogDestination(google::GLOG_INFO, "./rdma_info");  
  gflags::SetUsageMessage("Usage: ./rdma_network_test");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  LOG(INFO) << "Network Starting Up..." << endl;
  Network* net = oneflow::GetRdmaInstance();
  LOG(INFO) << "Create Rdma Instance Success." << endl;

  struct NetworkTopology net_topo;
  net_topo.all_nodes.resize(2);
  net_topo.all_nodes[FLAGS_my_machine_id].machine_id = FLAGS_my_machine_id;
  net_topo.all_nodes[FLAGS_my_machine_id].address = FLAGS_my_ip;
  net_topo.all_nodes[FLAGS_my_machine_id].port = 53433;
  net_topo.all_nodes[FLAGS_my_machine_id].neighbors.insert(FLAGS_peer_machine_id);
  net_topo.all_nodes[FLAGS_peer_machine_id].machine_id = FLAGS_peer_machine_id;
  net_topo.all_nodes[FLAGS_peer_machine_id].address = FLAGS_peer_ip;
  net_topo.all_nodes[FLAGS_peer_machine_id].port = 53433;
  net_topo.all_nodes[FLAGS_peer_machine_id].neighbors.insert(FLAGS_my_machine_id);
  
  // modify here manually
  int64_t my_machine_id = FLAGS_my_machine_id;
  int64_t peer_machine_id = FLAGS_peer_machine_id;

  net->Init(my_machine_id, net_topo);
  LOG(INFO) << "Net Init Success." << endl;

  NetworkMessage msg;
  NetworkResult result;
   
  /* 
  msg.src_machine_id = my_machine_id;
  msg.type = NetworkMessageType::kBarrier;
  msg.dst_machine_id = peer_machine_id;

  net->SendMsg(msg);
  cout << "PostSendRequest" << endl;

  int k = 0;
  for (int i = 0; i < 2 * net_topo.all_nodes[my_machine_id].neighbors.size(); ++i) {
    while (!net->Poll(&result)) {
#ifdef WIN32
      Sleep(1000);
#else
      sleep(1);
#endif
      printf("Poll time: %d, false\n", k++);
    }
    printf("Poll time: %d, true \n", k++);
    if (result.type == NetworkResultType::kSendOk) {
      printf("Send to %d OK\n", i);
    }
    else if (result.type == NetworkResultType::kReceiveMsg) {
      printf("Receive from %ld OK\n", result.net_msg.src_machine_id);
    }
    else {
      printf("Unexpected net event polled\n");
    }
  }

  cout << "Send/Recv test success." << endl;
  */

  
  clock_t start_time, current_time;

  // useful for my_machine_id == 0
  char* dst_buffer = new char[FLAGS_transfer_size];
  NetworkMemory* dst_memory = net->RegisterMemory(dst_buffer, FLAGS_transfer_size);
  MemoryDescriptor* remote_memory_descriptor = new MemoryDescriptor();
  remote_memory_descriptor->address = 0;

  // useful for my_machine_id == 1
  char* src_buffer = new char[FLAGS_transfer_size];
  NetworkMemory* src_memory = net->RegisterMemory(src_buffer, FLAGS_transfer_size);
  // send memory descriptor to peer
  if (my_machine_id == 1) {
    NetworkMessage memory_msg;
    memory_msg.type = NetworkMessageType::kRemoteMemoryDescriptor;
    memory_msg.src_machine_id = my_machine_id;
    memory_msg.dst_machine_id = peer_machine_id;
    memory_msg.address = src_memory->memory_discriptor().address;
    memory_msg.token = src_memory->memory_discriptor().remote_token;
    net->SendMsg(memory_msg);
  }
  
  // useful for all machine
  int i = 0;
  while (i < FLAGS_transfer_times) {
    while (!net->Poll(&result)) {
#ifdef WIN32
      Sleep(1000);
#else
      sleep(1);
#endif
      cout << "Poll result false" << endl;
    }
    if (result.type == NetworkResultType::kSendOk) {
      LOG(INFO) << "send ok" << endl;
    }
    else if (result.type == NetworkResultType::kReceiveMsg) {
      if (result.net_msg.type == NetworkMessageType::kRemoteMemoryDescriptor) {
        LOG(INFO) << "recv descriptor" << endl;
        remote_memory_descriptor->machine_id = result.net_msg.src_machine_id;
        remote_memory_descriptor->address = result.net_msg.address;
        remote_memory_descriptor->remote_token = result.net_msg.token;
        if (remote_memory_descriptor->address == 0) { 
          LOG(INFO) << "address error" << endl; 
          exit(1); 
        }
        std::cout << "remote_machine_id: "
                  << remote_memory_descriptor->machine_id
                  << ", remote_address: "
                  << remote_memory_descriptor->address
                  << ", remote_token: "
                  << remote_memory_descriptor->remote_token << std::endl;
        std::cout << "before post read" << std::endl;
        net->Read(*remote_memory_descriptor, dst_memory, [](){});
        LOG(INFO) << "async read issued" << endl;
        start_time = clock();
      }
      else if (result.net_msg.type == NetworkMessageType::kRequestAck) {
        LOG(INFO) << "Send next memory descriptor" << endl;
        NetworkMessage memory_msg;
        memory_msg.type = NetworkMessageType::kRemoteMemoryDescriptor;
        memory_msg.src_machine_id = my_machine_id;
        memory_msg.dst_machine_id = peer_machine_id;
        memory_msg.address = src_memory->memory_discriptor().address;
        memory_msg.token = src_memory->memory_discriptor().remote_token;
        net->SendMsg(memory_msg);
      }
    }
    else if (result.type == NetworkResultType::kReadOk) {
      current_time = clock();
      LOG(INFO) << "READ OK. TIMES: " << i 
        << ", cost time: " << (double)(current_time - start_time)/CLOCKS_PER_SEC 
        << endl;
      start_time = current_time;
      NetworkMessage read_ok_msg;
      read_ok_msg.type = NetworkMessageType::kRequestAck;
      read_ok_msg.src_machine_id = my_machine_id;
      read_ok_msg.dst_machine_id = peer_machine_id;
      net->SendMsg(read_ok_msg);
      ++i;
    }
  }

  delete []src_buffer;
  delete []dst_buffer;

  LOG(INFO) << "Network Shutting Down..." << endl;
  gflags::ShutDownCommandLineFlags();
  google::ShutdownGoogleLogging();
  return 0;
}
