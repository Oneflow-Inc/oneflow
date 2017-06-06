#include "network/rdma/linux/rdma_manager.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ctime>
#include <string>
#include "network/rdma/linux/interface.h"
#include "network/rdma/linux/connection.h"

namespace oneflow {

namespace {

struct sockaddr_in GetAddress(const char* ip, int port) {
  struct sockaddr_in addr = sockaddr_in();
  memset(&addr, 0, sizeof(sockaddr_in));
  inet_pton(AF_INET, ip, &addr.sin_addr);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  return addr;
}

}  // namespace

RdmaManager::RdmaManager() {
  context_ = NULL;
  listener_ = NULL;
  send_cq_ = NULL;
  recv_cq_ = NULL;
  protect_domain_ = NULL;
}

RdmaManager::~RdmaManager() {
}

bool RdmaManager::Init(const char* ip, int port) {
  my_addr_ = GetAddress(ip, port);

  // Init Adapter
  struct ibv_device** device_list = ibv_get_device_list(NULL);
  struct ibv_device* device = device_list[0];
  context_ = ibv_open_device(device);
  protect_domain_ = ibv_alloc_pd(context_);

  // Init env
  send_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe
  recv_cq_ = ibv_create_cq(context_, 10, NULL, NULL, 0);  // cqe

  my_sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

  int ret = bind(my_sock_, (struct sockaddr*)&my_addr_, sizeof(my_addr_));
  if (ret != 0)
    return false;

  ret = listen(my_sock_, 100);  // TODO(shiyuan) backlog
  if (ret != 0)
    return false;

  return true;
}

bool RdmaManager::Destroy() {
  // TODO(shiyuan)
  return true;
}

bool RdmaManager::CreateConnector(Connection* conn) {
  struct ibv_port_attr attr;
  int ret = ibv_query_port(context_, (uint8_t)1, &attr);
  if (ret != 0)
    return false;

  struct Connector* connector = new Connector;
  srand((unsigned)time(NULL));
  connector->my_lid = attr.lid;
  connector->my_qpn = 0;  // Will be set up after the creation of the queue pair
  connector->my_psn = static_cast<uint32_t>(rand()) & 0xffffff;
  union ibv_gid gid;
  ibv_query_gid(context_, (uint8_t)1, 0, &gid);
  connector->my_snp = gid.global.subnet_prefix;
  connector->my_iid = gid.global.interface_id;
  connector->active_mtu = attr.active_mtu;

  conn->set_connector(connector);

  return true;
}

bool RdmaManager::CreateQueuePair(Connection* conn) {
  struct ibv_qp_init_attr qp_init_attr;

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_context = NULL;
  qp_init_attr.send_cq = send_cq_;
  qp_init_attr.recv_cq = recv_cq_;
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.srq = NULL;
  qp_init_attr.sq_sig_all = 1;

  qp_init_attr.cap.max_send_wr = 10;
  qp_init_attr.cap.max_recv_wr = 10;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;

  struct ibv_qp* queue_pair = ibv_create_qp(protect_domain_, &qp_init_attr);;
  conn->set_queue_pair(queue_pair);

  if (queue_pair != NULL)
    return true;
  else
    return false;
}

RdmaMemory* RdmaManager::NewNetworkMemory() {
  // XXX(shiyuan)
  struct ibv_mr* memory_region = NULL;
  // HRESULT hr = adapter_->CreateMemoryRegion(
  //     IID_IND2MemoryRegion,
  //     overlapped_file_,
  //     reinterpret_cast<void**>(&memory_region));

  RdmaMemory* rdma_memory = new RdmaMemory(memory_region, protect_domain_);
  return rdma_memory;
}

uint64_t RdmaManager::WaitForConnection(Connection* conn,
                                        Request* recv_request) {
  uint64_t peer_machine_id;
  struct Connector* connector = conn->connector();
  struct Connector temp_connector;
  int read_bytes = 0;
  int total_read_bytes = 0;
  struct sockaddr_in peer_addr;
  socklen_t addr_len = sizeof(peer_addr);

  int peer_sock = accept(my_sock_, (struct sockaddr*)&peer_addr, &addr_len);

  read(peer_sock, &peer_machine_id, sizeof(uint64_t));

  write(peer_sock, connector, sizeof(struct Connector));

  total_read_bytes = 0;
  read_bytes = 0;
  while (total_read_bytes < sizeof(struct Connector)) {
    read_bytes = read(peer_sock, &temp_connector, sizeof(struct Connector));
    if (read_bytes > 0) {
      total_read_bytes += read_bytes;
    }
  }


  connector->peer_lid = temp_connector.my_lid;
  connector->peer_qpn = temp_connector.my_qpn;
  connector->peer_psn = temp_connector.my_psn;
  connector->peer_snp = temp_connector.my_snp;
  connector->peer_iid = temp_connector.my_iid;
  conn->set_connector(connector);

  close(peer_sock);

  conn->AcceptConnect();
  return peer_machine_id;
}

// |result| is owned by the caller, and the received message will be held in
// result->net_msg, having result->type == NetworkResultType::NET_RECEIVE_MSG.
int32_t RdmaManager::PollRecvQueue(NetworkResult* result) {
  struct ibv_wc wc;
  int32_t len = ibv_poll_cq(recv_cq_, 1, &wc);
  if (len <= 0) {  // return number of CQEs in array wc or -1 on error
    return -1;
  }
  if (wc.status != IBV_WC_SUCCESS) {
    return -1;
  }
  result->type = NetworkResultType::NET_RECEIVE_MSG;
  int32_t time_stamp = wc.wr_id;
  return time_stamp;
}

int32_t RdmaManager::PollSendQueue(NetworkResult* result) {
  struct ibv_wc wc;
  int32_t len = ibv_poll_cq(send_cq_, 1, &wc);

  if (len <= 0)  // return number of CQEs in array wc or -1 on error
    return -1;

  if (wc.status != IBV_WC_SUCCESS)
    return -1;

  switch (wc.opcode) {
    case IBV_WC_SEND: {
      result->type = NetworkResultType::NET_SEND_OK;
      // The context is the message timestamp in Send request.
      // Tehe network object does not have additional information
      // to convey to outside caller, it just recycle the
      // registered_message used in sending out.
      int32_t time_stamp = wc.wr_id;
      return time_stamp;
    }
    case IBV_WC_RDMA_READ: {
      result->type = NetworkResultType::NET_READ_OK;
      // The context is the message timestamp in Read request.
      // The network object needs to convey the information about
      // "what data have been read" to external caller.
      // The context is the message
      int32_t time_stamp = wc.wr_id;
      return time_stamp;
    }
    default:
      return -1;
  }
}

}  // namespace oneflow
