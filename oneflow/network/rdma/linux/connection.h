#ifndef ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_
#define ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <rdma/rdma_cma.h>

namespace oneflow{

class Connection {
public:
    Connection();
    ~Connection();
    Connection(int32_t peer_rank);

private:
    int32_t peer_rank_ = { -1 };
    
    struct ibv_qp* queue_pair_;

    struct ibv_mr* recv_mr_;
    struct ibv_mr* send_mr_;

    char* recv_region_;
    char* send_region_;

    // map peer_rank to rdma_cm_id
    
    void BuildConnection(struct rdma_cm_id* id);
    void BuildContext(struct ibv_context* verbs);
    void BuildQPAttr(struct ibv_qp_init_attr* queue_pair_attr);
    void BuildParams(struct rdma_comm_param* params);
    // void RegisterMemory();
    // void PostReceiver();
    // void 
};

} // namespace oneflow

#endif // ONEFLOW_NETWORK_RDMA_LINUX_CONNECTION_H_ 
