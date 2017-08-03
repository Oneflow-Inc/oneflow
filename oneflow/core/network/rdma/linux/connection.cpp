#include "oneflow/core/network/rdma/linux/connection.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "oneflow/core/network/rdma/linux/interface.h"
#include "oneflow/core/network/rdma/request_pool.h"

namespace oneflow {

namespace {

sockaddr_in GetAddress(const char* ip, int32_t port) {
  sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip, &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

void TransQueuePairState(
    const Connector& connector, ibv_qp* queue_pair) {
  ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(ibv_qp_attr));

  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = 1;
  qp_attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

  CHECK_EQ(ibv_modify_qp(queue_pair, &qp_attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                         IBV_QP_ACCESS_FLAGS),
           0);

  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = connector.active_mtu;
  qp_attr.dest_qp_num = connector.peer_qpn;
  qp_attr.rq_psn = connector.peer_psn;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  qp_attr.ah_attr.is_global = 1;
  qp_attr.ah_attr.grh.dgid.global.subnet_prefix = connector.peer_snp;
  qp_attr.ah_attr.grh.dgid.global.interface_id = connector.peer_iid;
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.hop_limit = 255;
  qp_attr.ah_attr.dlid = connector.peer_lid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = 1;

  CHECK_EQ(ibv_modify_qp(queue_pair, &qp_attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
      0);

  memset(&qp_attr, 0, sizeof(ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.sq_psn = connector.my_psn;
  qp_attr.timeout = 14;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.max_rd_atomic = 1;

  CHECK_EQ(ibv_modify_qp(queue_pair, &qp_attr,
                         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                         IBV_QP_MAX_QP_RD_ATOMIC),
           0);
}

}  // namespace

Connection::Connection(int64_t my_machine_id)
    : Connection::Connection(my_machine_id, -1) {}

Connection::Connection(int64_t my_machine_id, int64_t peer_machine_id)
    : my_machine_id_(my_machine_id),
      peer_machine_id_(peer_machine_id),
      connector_(nullptr),
      queue_pair_(nullptr) {}

Connection::~Connection() {
  DestroyConnection();
}

void Connection::set_connector(Connector* connector) {
  CHECK(!connector_);
  connector_ = connector;
}

void Connection::set_queue_pair(ibv_qp* queue_pair) {
  CHECK(!queue_pair_);
  queue_pair_ = queue_pair;
}

void Connection::Bind(const char* my_ip, int32_t my_port) {
  my_addr_ = GetAddress(my_ip, my_port);
  my_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  CHECK_EQ(bind(my_sock_, (struct sockaddr*)&my_addr_, sizeof(my_addr_)), 0);
}

bool Connection::TryConnectTo(const char* peer_ip, int32_t peer_port) {
  sockaddr_in peer_addr = GetAddress(peer_ip, peer_port);
  Connector temp_connector;
  int64_t read_bytes = 0;
  int64_t total_read_bytes = 0;
  int64_t rc = 0;
  int32_t peer_sock = socket(AF_INET, SOCK_STREAM, 0);
  int32_t ret = connect(peer_sock, reinterpret_cast<sockaddr*>(&peer_addr),
                        sizeof(peer_addr));
  if ((ret != 0) || (peer_sock < 0)) {
    CHECK_EQ(close(peer_sock), 0);
    return false;
  }

  rc = write(peer_sock, &my_machine_id_, sizeof(my_machine_id_));
  if (rc < sizeof(my_machine_id_)) {
    return false;
  } else {
    rc = 0;
  }

  while (!rc && total_read_bytes < sizeof(Connector)) {
    read_bytes = read(peer_sock, &temp_connector, sizeof(Connector));
    if (read_bytes > 0)
      total_read_bytes += read_bytes;
    else
      rc = read_bytes;
  }

  rc = write(peer_sock, connector_, sizeof(Connector));
  if (rc < sizeof(Connector)) {
    return false;
  } else {
    rc = 0;
  }

  connector_->peer_lid = temp_connector.my_lid;
  connector_->peer_qpn = temp_connector.my_qpn;
  connector_->peer_psn = temp_connector.my_psn;
  connector_->peer_snp = temp_connector.my_snp;
  connector_->peer_iid = temp_connector.my_iid;

  CHECK_EQ(close(peer_sock), 0);
  return true;
}

void Connection::CompleteConnection() {
  TransQueuePairState(*connector_, queue_pair_);
}

void Connection::AcceptConnect() {
  TransQueuePairState(*connector_, queue_pair_);
}

void Connection::Destroy() {
  delete connector_;
  connector_ = nullptr;
  if (queue_pair_ != nullptr) {
    CHECK_EQ(ibv_destroy_qp(queue_pair_), 0);
  }
}

void Connection::PostSendRequest(const Request& send_request) {
  ibv_send_wr wr;
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(&send_request);
  wr.next = nullptr;
  wr.sg_list =
      static_cast<ibv_sge*>(send_request.rdma_msg->net_memory()->sge());
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  CHECK_EQ(ibv_post_send(queue_pair_, &wr, &bad_wr), 0);
}

void Connection::PostRecvRequest(const Request& recv_request) {
  ibv_recv_wr wr;
  ibv_recv_wr* bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(&recv_request);
  wr.next = nullptr;
  wr.sg_list =
      static_cast<ibv_sge*>(recv_request.rdma_msg->net_memory()->sge());
  wr.num_sge = 1;
  CHECK_EQ(ibv_post_recv(queue_pair_, &wr, &bad_wr), 0);
}

void Connection::PostReadRequest(
    const Request& read_request,
    const MemoryDescriptor& remote_memory_descriptor,
    RdmaMemory* dst_memory) {
  ibv_send_wr wr;
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = reinterpret_cast<uint64_t>(&read_request);
  wr.opcode = IBV_WR_RDMA_READ;
  wr.sg_list = static_cast<ibv_sge*>(dst_memory->sge());
  wr.num_sge = 1;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = remote_memory_descriptor.address;
  wr.wr.rdma.rkey = remote_memory_descriptor.remote_token;

  CHECK_EQ(ibv_post_send(queue_pair_, &wr, &bad_wr), 0);
}

}  // namespace oneflow
