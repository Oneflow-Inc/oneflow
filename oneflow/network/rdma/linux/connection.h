#ifndef ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

namespace oneflow{

class Request;
class Memory;

extern const int BUFFER_SIZE;

struct Context {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_comp_channel *comp_channel;

  pthread_t cq_poller_thread;
};

// Response for build connection.
// Connect to other machine or connected by other machine.
//
class Connection {
 public:
  Connection();
  explicit Connection(uint64_t peer_machine_id);
  ~Connection();
  bool Bind();
  bool TryConnectTo();
  void CompleteConnectionTo();
  void AcceptConnect();

  void DestroyConnection();

  void PostSendRequest(Request* send_request);
  void PostRecvRequest(Request* recv_request);
  void PostReadRequest(Request* read_request,
                       MemoryDescriptor* remote_memory_descriptor,
                       Memory* dst_memory);

  connector; // TODO(shiyuan)
  struct ibv_qp* queue_pair;

 private:
  uint64_t my_machine_id_;
  uint64_t peer_machine_id_ = { 0 };
  
  struct rdma_cm_id* id_;

  struct ibv_mr* recv_mr_;
  struct ibv_mr* send_mr_;

  char* recv_region_;
  char* send_region_;
  
  struct ibv_qp* qp;
  struct Context *s_ctx;

  int32_t num_completions;
  addr my_sock_, peer_sock_;

  void BuildConnection();
  void BuildContext(struct ibv_context* verbs);
  void BuildQPAttr(struct ibv_qp_init_attr* queue_pair_attr);
  void BuildParams(struct rdma_comm_param* params);
  
  int OnEvent(struct rdma_cm_event* event);
  void RegisterMemory();
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_ 
