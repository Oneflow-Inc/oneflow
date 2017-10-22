#include "oneflow/core/comm_network/rdma/endpoint_manager.h"
#include <arpa/inet.h>
#include "glog/logging.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const std::string& ip, int32_t port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

void EndpointManager::Init(const std::string& my_ip, int32_t my_port) {
  // Init Adapter
  ibv_device** device_list = ibv_get_device_list(NULL);
  ibv_device* device = device_list[0];
  context_ = ibv_open_device(device);
  CHECK(context_);
  pd_ = ibv_alloc_pd(context_);
  CHECK(pd_);

  // Init env
  send_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(send_cq_);
  recv_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(recv_cq_);

  ibv_free_device_list(device_list);
}

RdmaMem* EndpointManager::NewRdmaMem() {
  RdmaMem* rdma_mem = new RdmaMem(pd_);
  CHECK(rdma_mem);
  return rdma_mem;
}

Connector* EndpointManager::NewConnector() {
  ibv_port_attr attr;
  CHECK_EQ(ibv_query_port(context_, (uint8_t)1, &attr), 0);

  Connector* connector = new Connector;
  CHECK(connector);
  srand((unsigned)time(NULL));
  connector->my_conn_info.lid = attr.lid;
  connector->my_conn_info.qpn =
      0;  // Will be set up after the creation of the queue pair
  connector->my_conn_info.snp = static_cast<uint32_t>(rand()) & 0xffffff;
  union ibv_gid gid;
  CHECK_EQ(ibv_query_gid(context_, (uint8_t)1, 0, &gid), 0);
  connector->my_conn_info.snp = gid.global.subnet_prefix;
  connector->my_conn_info.iid = gid.global.interface_id;
  connector->active_mtu = attr.active_mtu;
  return connector;
}

ibv_qp* EndpointManager::NewQueuePair() {
  ibv_qp_init_attr qp_init_attr;

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_context = nullptr;
  qp_init_attr.send_cq = send_cq_;
  qp_init_attr.recv_cq = recv_cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.srq = nullptr;
  qp_init_attr.sq_sig_all = 1;

  qp_init_attr.cap.max_send_wr = 10;
  qp_init_attr.cap.max_recv_wr = 10;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  ibv_qp* queue_pair = ibv_create_qp(pd_, &qp_init_attr);
  CHECK(queue_pair);
  return queue_pair;
}

}  // namespace oneflow
