#ifndef ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H
#define ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H

namespace oneflow {

class EndpointManager {
public:
  void Start();

private:
  void PollLoop();
  bool PollSendQueue();
  bool PollRecvQueue();
};

} // oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_RDMA_LINUX_ENDPOINT_MANAGER_H
