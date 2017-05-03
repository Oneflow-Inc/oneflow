#ifndef ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_

#include <stdio.h>
#include <rdma/rdma_cma.h>

namespace oneflow{

/* 
 * TODO(shiyuan): Need mv connection to Class Connection
 * struct connection {
 *   struct rdma_cm_id* id;
 *   struct ibv_qp* qp;
 *   
 *   struct ibv_mr* recv_mr;
 *   struct ibv_mr* send_mr;
 *
 *   char* recv_region;
 *   char* send_region;
 *
 *   int num_completions;
 * };
 *
 */


class RdmaManager {

public:
  struct rdm_cm_id* id;
  struct addrinfo* addr;


  RdmaManager();
  ~RdmaManager();

  bool Init();
  bool Destroy();
  bool GetAddress(string addr, string port);

private:
  struct rdma_cm_id* id_;
  struct addrinfo* addr;
  struct ibv_context ctx_;
  struct rdma_cm_event* event_;
  struct rdma_event_channel* ec_;
  struct ibv_pd* pd_;
  struct ibv_cq* recv_cq_;
  struct ibv_cq* send_cq_;
  struct ibv_comp_channel* comp_channel_;







  bool InitDevice();
  bool InitAdapter();
  bool InitEnv();
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_RDMA_MANAGER_H_
