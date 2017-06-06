#include "network/rdma/linux/connection.h"
#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <iostream>
#include "network/rdma/linux/interface.h"
#include "network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const char* ip, int port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip, &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

Connection::Connection(uint64_t my_machine_id)
    : Connection::Connection(my_machine_id, -1) {}

Connection::Connection(uint64_t my_machine_id, uint64_t peer_machine_id) {
  my_machine_id_ = my_machine_id;
  peer_machine_id_ = peer_machine_id;
  connector_ = NULL;
  queue_pair_ = NULL;
}

Connection::~Connection() {
}

bool Connection::Bind(const char* ip, int port) {
  my_addr_ = GetAddress(ip, port);
  my_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  int ret = bind(my_sock_, (struct sockaddr*)&my_addr_, sizeof(my_addr_));
  if (ret == 0)
    return true;
  else
    return false;
}

bool Connection::TryConnectTo(const char* peer_ip, int port) {
  struct sockaddr_in peer_addr = GetAddress(peer_ip, port);
  struct Connector temp_connector;
  int read_bytes = 0;
  int total_read_bytes = 0;
  int rc;
  int peer_sock = socket(AF_INET, SOCK_STREAM, 0);
  int ret = connect(peer_sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr));
  if ((ret != 0) || (peer_sock < 0)) {
    close(peer_sock);
    return false;
  }

  rc = write(peer_sock, &my_machine_id_, sizeof(my_machine_id_));
  if (rc < sizeof(my_machine_id_))
    return false;
  else
    rc = 0;

  while (!rc && total_read_bytes < sizeof(struct Connector)) {
    read_bytes  = read(peer_sock, &temp_connector, sizeof(struct Connector));
    if (read_bytes > 0)
      total_read_bytes += read_bytes;
    else
      rc = read_bytes;
  }

  rc = write(peer_sock, connector_, sizeof(struct Connector));
  if (rc < sizeof(struct Connector))
    return false;
  else
    rc = 0;

  connector_->peer_lid = temp_connector.my_lid;
  connector_->peer_qpn = temp_connector.my_qpn;
  connector_->peer_psn = temp_connector.my_psn;
  connector_->peer_snp = temp_connector.my_snp;
  connector_->peer_iid = temp_connector.my_iid;

  ret = close(peer_sock);
  if (ret != 0)
    return false;

  return true;
}

void Connection::TransferQueuePair() {
  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;
  qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                            IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_READ;

  ibv_modify_qp(queue_pair_, &qp_attr,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = connector_->active_mtu;
  qp_attr.dest_qp_num = connector_->peer_qpn;
  qp_attr.rq_psn = connector_->peer_psn;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.grh.dgid.global.subnet_prefix = connector_->peer_snp;
  qp_attr.ah_attr.grh.dgid.global.interface_id = connector_->peer_iid;
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.hop_limit = 255;
  qp_attr.ah_attr.dlid = connector_->peer_lid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = 1;

  ibv_modify_qp(queue_pair_, &qp_attr,
                IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER);

  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = connector_->my_psn;
  qp_attr.timeout = 14;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.max_rd_atomic = 1;

  ibv_modify_qp(queue_pair_, &qp_attr,
                IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
}

void Connection::CompleteConnectionTo() {
  TransferQueuePair();
}

void Connection::AcceptConnect() {
  TransferQueuePair();
}

void Connection::DestroyConnection() {
}

void Connection::PostSendRequest(Request* send_request) {
  struct ibv_send_wr wr, *bad_wr = NULL;
  wr.wr_id = send_request->time_stamp;
  wr.next = NULL;
  wr.sg_list = static_cast<ibv_sge*>(
      send_request->rdma_msg->net_memory()->sge());
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  ibv_post_send(queue_pair_, &wr, &bad_wr);
}

void Connection::PostRecvRequest(Request* recv_request) {
  struct ibv_recv_wr wr, *bad_wr = NULL;
  wr.wr_id = recv_request->time_stamp;
  wr.next = NULL;
  wr.sg_list = static_cast<ibv_sge*>(
      recv_request->rdma_msg->net_memory()->sge());
  wr.num_sge = 1;
  ibv_post_recv(queue_pair_, &wr, &bad_wr);
}

void Connection::PostReadRequest(
    Request* read_request,
    MemoryDescriptor* remote_memory_descriptor,
    RdmaMemory* dst_memory) {
  struct ibv_send_wr wr, *bad_wr = NULL;
  wr.wr_id = read_request->time_stamp;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = static_cast<ibv_sge*>(dst_memory->sge());
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_memory_descriptor->address;
  wr.wr.rdma.rkey = remote_memory_descriptor->remote_token;

  ibv_post_send(queue_pair_, &wr, &bad_wr);
}

}  // namespace oneflow
