#include "oneflow/core/network/rdma/linux/rdma_manager.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <ctime>
#include <string>
#include "oneflow/core/network/rdma/linux/connection.h"
#include "oneflow/core/network/rdma/linux/interface.h"

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

RdmaManager::~RdmaManager() {
  Destroy();
}

void RdmaManager::Init(const char* my_ip, int32_t my_port) {
  my_addr_ = GetAddress(my_ip, my_port);

  // Init Adapter
  ibv_device** device_list = ibv_get_device_list(NULL);
  ibv_device* device = device_list[0];
  context_ = ibv_open_device(device);
  CHECK(context_);
  protect_domain_ = ibv_alloc_pd(context_);
  CHECK(protect_domain_);

  // Init env
  send_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(send_cq_);
  recv_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  CHECK(recv_cq_);

  my_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  CHECK(my_sock_);

  CHECK_EQ(bind(my_sock_, (sockaddr*)&my_addr_, sizeof(my_addr_)), 0);

  CHECK_EQ(listen(my_sock_, 100), 0);  // TODO(shiyuan) backlog
  ibv_free_device_list(device_list);
  device_list = nullptr;
}

void RdmaManager::Destroy() {
  if (send_cq_ != nullptr) {
    CHECK_EQ(ibv_destroy_cq(send_cq_), 0);
  }
  if (recv_cq_ != nullptr) {
    CHECK_EQ(ibv_destroy_cq(recv_cq_), 0);
  }

  if (protect_domain_ != nullptr) {
    CHECK_EQ(ibv_dealloc_pd(protect_domain_), 0);
  }

  if (context_ != nullptr) {
    CHECK_EQ(ibv_close_device(context_), 0);
  }
}

void RdmaManager::CreateConnector(Connection* conn) {
  ibv_port_attr attr;
  CHECK_EQ(ibv_query_port(context_, (uint8_t)1, &attr), 0);

  Connector* connector = new Connector;
  CHECK(connector);
  srand((unsigned)time(NULL));
  connector->my_lid = attr.lid;
  connector->my_qpn = 0;  // Will be set up after the creation of the queue pair
  connector->my_psn = static_cast<uint32_t>(rand()) & 0xffffff;
  union ibv_gid gid;
  CHECK_EQ(ibv_query_gid(context_, (uint8_t)1, 0, &gid), 0);
  connector->my_snp = gid.global.subnet_prefix;
  connector->my_iid = gid.global.interface_id;
  connector->active_mtu = attr.active_mtu;

  conn->set_connector(connector);
}

void RdmaManager::CreateQueuePair(Connection* conn) {
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

  ibv_qp* queue_pair = ibv_create_qp(protect_domain_, &qp_init_attr);
  CHECK(queue_pair);
  conn->set_queue_pair(queue_pair);
}

RdmaMemory* RdmaManager::NewNetworkMemory() {
  RdmaMemory* rdma_memory = new RdmaMemory(protect_domain_);
  CHECK(rdma_memory);
  return rdma_memory;
}

int64_t RdmaManager::WaitForConnection(Connection* conn,
                                       Request* recv_request) {
  int64_t peer_machine_id;
  Connector* connector = conn->mutable_connector();
  Connector temp_connector;
  int read_bytes = 0;
  int total_read_bytes = 0;
  sockaddr_in peer_addr;
  socklen_t addr_len = sizeof(peer_addr);

  int peer_sock = accept(my_sock_, (sockaddr*)&peer_addr, &addr_len);
  CHECK_NE(peer_sock, -1);
  CHECK_GT(read(peer_sock, &peer_machine_id, sizeof(int64_t)), 0);
  CHECK_GT(write(peer_sock, connector, sizeof(Connector)), 0);

  total_read_bytes = 0;
  read_bytes = 0;
  while (total_read_bytes < sizeof(Connector)) {
    read_bytes = read(peer_sock, &temp_connector, sizeof(Connector));
    if (read_bytes > 0) { total_read_bytes += read_bytes; }
  }

  connector->peer_lid = temp_connector.my_lid;
  connector->peer_qpn = temp_connector.my_qpn;
  connector->peer_psn = temp_connector.my_psn;
  connector->peer_snp = temp_connector.my_snp;
  connector->peer_iid = temp_connector.my_iid;

  CHECK_EQ(close(peer_sock), 0);

  conn->AcceptConnect();
  return peer_machine_id;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::kReceiveMsg.
Request* RdmaManager::PollRecvQueue(NetworkResult* result) {
  ibv_wc wc;
  int32_t len = ibv_poll_cq(recv_cq_, 1, &wc);

  // return number of CQEs in array wc or -1 on error
  if (len <= 0) { return nullptr; }

  CHECK_EQ(wc.status, IBV_WC_SUCCESS);

  result->type = NetworkResultType::kReceiveMsg;
  return reinterpret_cast<Request*>(wc.wr_id);
}

Request* RdmaManager::PollSendQueue(NetworkResult* result) {
  ibv_wc wc;
  int32_t len = ibv_poll_cq(send_cq_, 1, &wc);

  // return number of CQEs in array wc or -1 on error
  if (len <= 0) { return nullptr; }

  CHECK_EQ(wc.status, IBV_WC_SUCCESS);

  switch (wc.opcode) {
    case IBV_WC_SEND: {
      result->type = NetworkResultType::kSendOk;
      // The context is the message timestamp in Send request.
      // Tehe network object does not have additional information
      // to convey to outside caller, it just recycle the
      // registered_message used in sending out.
      return reinterpret_cast<Request*>(wc.wr_id);
    }
    case IBV_WC_RDMA_READ: {
      result->type = NetworkResultType::kReadOk;
      // The context is the message timestamp in Read request.
      // The network object needs to convey the information about
      // "what data have been read" to external caller.
      // The context is the message
      return reinterpret_cast<Request*>(wc.wr_id);
    }
    default: return nullptr;
  }
}

}  // namespace oneflow
