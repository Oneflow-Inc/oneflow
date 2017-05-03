#ifndef ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

namespace oneflow{

extern const int BUFFER_SIZE;

struct Context {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_comp_channel *comp_channel;

  pthread_t cq_poller_thread;
};

class Connection {
public:
  Connection();
  ~Connection();
  Connection(uint64_t peer_machine_id);
  void DestroyConnection();

private:
  uint64_t peer_machine_id_ = { -1 };
  
  struct rdma_cm_id* id_;
  struct ibv_qp* queue_pair_;

  struct ibv_mr* recv_mr_;
  struct ibv_mr* send_mr_;

  char* recv_region_;
  char* send_region_;
  
  struct Context *s_ctx;

  void BuildConnection();
  void BuildContext(struct ibv_context* verbs);
  void BuildQPAttr(struct ibv_qp_init_attr* queue_pair_attr);
  void BuildParams(struct rdma_comm_param* params);
  
  int OnEvent(struct rdma_cm_event* event);
  void RegisterMemory();
  
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_ 
