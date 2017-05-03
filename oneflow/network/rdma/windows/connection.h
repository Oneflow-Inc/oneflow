#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// #include <unistd.h>
#include <cstdint>
#include "network/rdma/windows/ndsupport.h"
#include "network/rdma/windows/interface.h"
#include "network/network_memory.h"


namespace oneflow {

class Request;
class Memory;

class Connection {
 public:
  Connection();
  ~Connection();
  explicit Connection(uint64_t peer_machine_id);

  void PostSendRequest(Request* send_request);
  void PostRecvRequest(Request* recv_request);
  void PostReadRequest(Request* read_request,
                       MemoryDescriptor* remote_memory_descriptor,
                       Memory* dst_memory);

  bool Bind();
  // connect to and connected
  bool TryConnectTo();
  void CompleteConnectionTo();
  // void WaitForConnection();
  void AcceptConnect();

  void DestroyConnection();

  IND2Connector* connector;
  IND2QueuePair* queue_pair;
  OVERLAPPED ov;

 private:
  uint64_t my_machine_id_;
  uint64_t peer_machine_id_{ 0 };  // TODO(shiyuan)

  sockaddr_in my_sock_, peer_sock_;

  // prepare for connect
  // set up parameters
  // void BuildContext();
  // void BuildParams();

  // destroy connect
};

}  // namespace oneflow

#endif  // ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
