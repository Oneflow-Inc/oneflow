#include "network/rdma/linux/connection.h"

namespace oneflow {

const int BUFFER_SIZE = 1024;

Connection::Connection() : Connection::Connection(-1) {}

Connection::Connection(uint64_t peer_machine_id)
{
    peer_machine_id_ = peer_machine_id;
    
    recv_mr_ = NULL;
    send_mr_ = NULL;

    recv_region_ = (char*)malloc(BUFFER_SIZE);
    send_region_ = (char*)malloc(BUFFER_SIZE);
    
    memset(recv_region_, 0, BUFFER_SIZE);
    memset(send_region_, 0, BUFFER_SIZE);
    
    s_ctx = NULL;
}

Connection::~Connection() 
{
    free(queue_pair_);
    free(recv_mr_);
    free(send_mr_);
    free(recv_region_);
    free(send_region_);
}

/*
void Connection::ReBuildConnection(uint64_t peer_machine_id)
{
    if (peer_machine_id_ == -1) {
        peer_machine_id_ = peer_machine_id;
        id = 

        BuildConnection(
*/

int Connection::OnEvent(struct rdma_cm_event *event)
{
    int r = 0;
    if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
        r = BuildConnection(event->id);
    }
    else if (event->event == RDMA_CM_EVENT_ESTABLISHED) {
        r = OnConnection(event->id);
    }
    else if (event->event == RDMA_CM_EVENT_DISCONNECTED) {
        r = DistroyConnection(event->id);
    }

    return r;
}



int Connection::BuildConnection(struct rdma_cm_id* id)
{
    //if peer_machine_id_ != -1 
    //    re build connection
    struct ibv_qp_init_attr qp_attr;
    struct rdma_conn_param cm_params;
    
    BuildContext(id->verbs);
    BuildQPAttr(&qp_attr);

    rdma_create_qp(id, s_ctx->pd, &qp_attr);

    id->context = ;// need update
    queue_pair_ = id->qp;

    register_memory();
    post_receives(); // need update

    memset(&cm_params, 0, sizeof(cm_params));
    rdma_accept(id, &cm_params);
    
    return 0;
}

void Connection::BuildContext(struct ibv_context *verbs)
{
    if (s_ctx_) {
        if (s_ctx_->ctx != verbs) {
            die("cannot handle events in more than one context.");
        }
        return ;
    }

    s_ctx_ = (struct Context*)malloc(sizeof(struct Context));
    s_ctx_->ctx = verbs;
    s_ctx_->pd = ibv_alloc_pd(s_ctx_->ctx);
    s_ctx_->comp_channel = ibv_create_comp_channel(s_ctx_->ctx);
    s_ctx_->cq = ibv_create_cq(s_ctx_->ctx, 10, NULL, s_ctx_->comp_channel, 0); // need update
    ibv_req_notify_cq(s_ctx_->cq, 0);

    pthread_create(&s_ctx_->cq_poller_thread, NULL, poll_cq, NULL); // need update
}

//void* poll_cq(void* ctx);

void Connection::BuildQPAttr(struct ibv_qp_init_attr* qp_attr)
{
    memset(qp_attr, 0, sizeof(*qp_attr));

    qp_attr->send_cq = s_ctx_->cq;
    qp_attr->recv_cq = s_ctx_->cq;
    qp_attr->qp_type = IBV_QPT_RC;

    qp_attr->cap.max_send_wr = 10; // need update
    qp_attr->cap.max_recv_wr = 10; //
    qp_attr->cap.max_send_sge = 1; // 
    qp_attr->cap.max_recv_sge = 1; // 

}

void Connection::RegisterMemory()
{
    send_mr_ = ibv_reg_mr(s_ctx->pd, send_region_, BUFFER_SIZE, 
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    
    recv_mr_ = ibv_reg_mr(s_ctx->pd, recv_region_, BUFFER_SIZE, 
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
}



void Connection::DestroyConnection()
{
    rdma_destroy_qp(id);

    ibv_dereg_mr(send_mr_);
    ibv_dereg_mr(recv_mr_);

    peer_machine_id_ = -1;
}



} // namespace oneflow

