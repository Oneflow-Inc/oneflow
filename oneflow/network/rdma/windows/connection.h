#ifndef ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// #include <unistd.h>
#include <cstdint>
#include "network/rdma/windows/ndsupport.h"

namespace oneflow {

class Connection {
public: 
  Connection();
  ~Connection();
  Connection(uint64_t peer_machine_id);

private:
  uint64_t peer_machine_id_{ -1 };

  IND2Connector* connector_;
  IND2QueuePair* queue_pair_;
  OVERLAPPED ov_;

  // prepare for connect
  // set up parameters
  void BuildContext();
  void BuildParams();

  // connect to and connected
  void TryConnectTo();
  void CompleteConnectionTo();
  void WaitForConnection();

  //void PostRecvRequest(); TODO(shiyuan) Not necessarily at this level

  // destroy connect
  void DestroyConnection();

};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_WINDOWS_CONNECTION_H_
