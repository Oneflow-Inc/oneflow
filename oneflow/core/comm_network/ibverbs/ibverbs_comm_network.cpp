#include "oneflow/core/comm_network/ibverbs/ibverbs_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"

#if defined(WITH_RDMA) && defined(PLATFORM_POSIX)

#include <fcntl.h>
#include <netdb.h>
#include <arpa/inet.h>

namespace oneflow {

namespace {

std::string GenTokensMsgKey(int64_t machine_id) {
  return "IBVerbsTokensMsg/" + std::to_string(machine_id);
}

std::string GenConnInfoKey(int64_t src_machine_id, int64_t dst_machine_id) {
  return "IBVerbsConnInfo/" + std::to_string(src_machine_id) + "/" + std::to_string(dst_machine_id);
}

int32_t NumOfActivePorts(ibv_device* device) {
  int32_t active_ports = 0;
  ibv_device_attr device_attr;
  ibv_port_attr port_attr;
  ibv_context* context = ibv_open_device(device);
  PCHECK(context);
  PCHECK(ibv_query_device(context, &device_attr) == 0);
  for (int32_t port_index = 1; port_index <= device_attr.phys_port_cnt; ++port_index) {
    PCHECK(ibv_query_port(context, port_index, &port_attr) == 0);
    if (port_attr.state == IBV_PORT_ACTIVE) { active_ports++; }
  }
  ibv_close_device(context);
  return active_ports;
}

int32_t ReadSysfsFile(const char* dir, const char* file, char* buf, size_t size) {
  char* path;
  int32_t fd;
  int32_t len;
  if (asprintf(&path, "%s/%s", dir, file) < 0) { return -1; }
  fd = open(path, O_RDONLY);
  if (fd < 0) {
    free(path);
    return -1;
  }
  len = read(fd, buf, size);
  close(fd);
  free(path);
  if (len > 0 && buf[len - 1] == '\n') { buf[--len] = '\0'; }
  return len;
}

bool IsGidTypeRoceV2(const char* dir, uint32_t port_num, uint32_t index) {
  char name[32];
  char buff[41];
  snprintf(name, sizeof(name), "ports/%d/gid_attrs/types/%d", port_num, index);
  if (ReadSysfsFile(dir, name, buff, sizeof(buff)) <= 0) { return false; }
  return !strcmp(buff, "RoCE v2");
}

}  // namespace

IBVerbsCommNet::~IBVerbsCommNet() {
  while (poll_exit_flag_.test_and_set() == true) {}
  poll_thread_.join();
  for (IBVerbsQP* qp : qp_vec_) {
    if (qp) { delete qp; }
  }
  PCHECK(ibv_destroy_cq(cq_) == 0);
  PCHECK(ibv_dealloc_pd(pd_) == 0);
  PCHECK(ibv_close_device(context_) == 0);
}

void IBVerbsCommNet::RegisterMemoryDone() {
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  IBVerbsTokensMsg this_tokens_msg;
  for (IBVerbsMemDesc* mem_desc : mem_descs()) {
    this_tokens_msg.mutable_token2mem_desc()->insert(
        {reinterpret_cast<uint64_t>(mem_desc), mem_desc->ToProto()});
  }
  Global<CtrlClient>::Get()->PushKV(GenTokensMsgKey(this_machine_id), this_tokens_msg);
  for (int64_t peer_mchn_id : peer_machine_id()) {
    IBVerbsTokensMsg peer_tokens_msg;
    Global<CtrlClient>::Get()->PullKV(GenTokensMsgKey(peer_mchn_id), &peer_tokens_msg);
    for (const auto& pair : peer_tokens_msg.token2mem_desc()) {
      CHECK(token2mem_desc_.at(peer_mchn_id)
                .emplace(reinterpret_cast<void*>(pair.first), pair.second)
                .second);
    }
  }
  OF_BARRIER();
  Global<CtrlClient>::Get()->ClearKV(GenTokensMsgKey(this_machine_id));
}

void IBVerbsCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) {
  qp_vec_.at(dst_machine_id)->PostSendRequest(msg);
}

void IBVerbsCommNet::InitContext(const std::string& device_name) {
  int32_t device_num = 0;
  int32_t device_index = 0;
  ibv_device** device_list = ibv_get_device_list(&device_num);
  PCHECK(device_list);
  ibv_device* device = nullptr;
  if (device_name == "") {
    for (device_index = 0; device_index < device_num; ++device_index) {
      device = device_list[device_index];
      if (NumOfActivePorts(device) > 0) { break; }
    }
    CHECK_LT(device_index, device_num) << "There is no device with active port in the machine.";
  } else {
    for (device_index = 0; device_index < device_num; ++device_index) {
      device = device_list[device_index];
      if (std::string(ibv_get_device_name(device)) == device_name) {
        CHECK(NumOfActivePorts(device)) << "Device " << device_name << " has no active port.";
        break;
      }
    }
    CHECK_LT(device_index, device_num) << "The device " << device_name << " wasn't found.";
  }
  PCHECK(device);
  context_ = ibv_open_device(device);
  PCHECK(context_);
  ibv_free_device_list(device_list);
}

uint32_t IBVerbsCommNet::QueryPort(uint32_t port_num, ibv_port_attr* port_attr) {
  ibv_device_attr device_attr;
  PCHECK(ibv_query_device(context_, &device_attr) == 0);
  if (port_num == 0) {
    for (size_t port_index = 1; port_index <= device_attr.phys_port_cnt; ++port_index) {
      PCHECK(ibv_query_port(context_, port_index, port_attr) == 0);
      if (port_attr->state == IBV_PORT_ACTIVE) {
        port_num = port_index;
        break;
      }
    }
    CHECK_GT(port_num, 0) << "No active ports";
  } else {
    CHECK_LE(port_num, device_attr.phys_port_cnt)
        << "DEVICE_PORT should be less or equal to amount of available ports";
    PCHECK(ibv_query_port(context_, port_num, port_attr) == 0);
    CHECK_EQ(port_attr->state, IBV_PORT_ACTIVE) << "Selected DEVICE_PORT is not active";
  }
  return port_num;
}

uint32_t IBVerbsCommNet::QueryGid(uint32_t port_num, uint32_t sgid_index, ibv_port_attr* port_attr,
                                  ibv_gid* gid) {
  uint32_t gids_num = 0;
  uint32_t v2_ip_num = 0;
  uint32_t gid_index = 0;
  for (int32_t i = 0; i < port_attr->gid_tbl_len; i++) {
    PCHECK(ibv_query_gid(context_, port_num, i, gid) == 0)
        << "Failed to query gid to port " << port_num << " index " << i;
    if (gid->global.interface_id) {
      gids_num++;
      if (gid->global.subnet_prefix == 0
          && IsGidTypeRoceV2(context_->device->ibdev_path, port_num, i)) {
        if (v2_ip_num == 0) { gid_index = i; }
        v2_ip_num++;
      }
    }
  }
  switch (port_attr->link_layer) {
    case (IBV_LINK_LAYER_ETHERNET):
      if (sgid_index != 0) {
        CHECK_LT(sgid_index, gids_num)
            << "RDMA_GID_INDEX should be less than GIDs amount" << gids_num;
        gid_index = sgid_index;
      }
      if (!IsGidTypeRoceV2(context_->device->ibdev_path, port_num, gid_index)) {
        LOG(INFO) << "RoCE v2 is not configured for GID_INDEX " << gid_index;
      }
      break;
    case (IBV_LINK_LAYER_INFINIBAND): break;
    default:
      LOG(INFO) << "Unknown port link layer. Currently supporting Ethernet and InfiniBand only.";
  }
  PCHECK(ibv_query_gid(context_, port_num, gid_index, gid) == 0);
  return gid_index;
}

IBVerbsCommNet::IBVerbsCommNet(const Plan& plan)
    : CommNetIf(plan),
      token2mem_desc_(Global<JobDesc>::Get()->TotalMachineNum()),
      poll_exit_flag_(ATOMIC_FLAG_INIT) {
  auto& ibv_conf = Global<JobDesc>::Get()->ibverbs_conf();
  InitContext(ibv_conf.device_name());
  pd_ = ibv_alloc_pd(context_);
  PCHECK(pd_);
  cq_ = ibv_create_cq(context_, max_poll_wc_num_, nullptr, nullptr, 0);
  PCHECK(cq_);

  ibv_port_attr port_attr;
  ibv_gid gid;
  uint32_t port_num = QueryPort(static_cast<uint32_t>(ibv_conf.port_num()), &port_attr);
  uint32_t sgid_index =
      QueryGid(port_num, static_cast<uint32_t>(ibv_conf.sgid_index()), &port_attr, &gid);
  auto mutable_ibv_conf = Global<JobDesc>::Get()->mutable_ibverbs_conf();
  mutable_ibv_conf->set_port_num(port_num);
  mutable_ibv_conf->set_sgid_index(sgid_index);

  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  qp_vec_.assign(Global<JobDesc>::Get()->TotalMachineNum(), nullptr);
  for (int64_t peer_mchn_id : peer_machine_id()) {
    IBVerbsQP* cur_qp = new IBVerbsQP(context_, pd_, cq_);
    qp_vec_.at(peer_mchn_id) = cur_qp;
    IBVerbsConnectionInfo conn_info;
    conn_info.set_local_id(port_attr.lid);
    conn_info.set_qp_num(cur_qp->qp_num());
    conn_info.set_subnet_prefix(gid.global.subnet_prefix);
    conn_info.set_interface_id(gid.global.interface_id);
    Global<CtrlClient>::Get()->PushKV(GenConnInfoKey(this_machine_id, peer_mchn_id), conn_info);
  }
  for (int64_t peer_mchn_id : peer_machine_id()) {
    IBVerbsConnectionInfo conn_info;
    Global<CtrlClient>::Get()->PullKV(GenConnInfoKey(peer_mchn_id, this_machine_id), &conn_info);
    qp_vec_.at(peer_mchn_id)->Connect(conn_info);
  }
  OF_BARRIER();
  for (int64_t peer_mchn_id : peer_machine_id()) {
    qp_vec_.at(peer_mchn_id)->PostAllRecvRequest();
    Global<CtrlClient>::Get()->ClearKV(GenConnInfoKey(this_machine_id, peer_mchn_id));
  }
  OF_BARRIER();
  poll_thread_ = std::thread(&IBVerbsCommNet::PollCQ, this);
  OF_BARRIER();
}

void IBVerbsCommNet::DoRead(void* read_id, int64_t src_machine_id, void* src_token,
                            void* dst_token) {
  qp_vec_.at(src_machine_id)
      ->PostReadRequest(token2mem_desc_.at(src_machine_id).at(src_token),
                        *static_cast<const IBVerbsMemDesc*>(dst_token), read_id);
}

void IBVerbsCommNet::PollCQ() {
  std::vector<ibv_wc> wc_vec(max_poll_wc_num_);
  while (poll_exit_flag_.test_and_set() == false) {
    poll_exit_flag_.clear();
    int32_t found_wc_num = ibv_poll_cq(cq_, max_poll_wc_num_, wc_vec.data());
    CHECK_GE(found_wc_num, 0);
    FOR_RANGE(int32_t, i, 0, found_wc_num) {
      const ibv_wc& wc = wc_vec.at(i);
      CHECK_EQ(wc.status, IBV_WC_SUCCESS) << "Failed status \n"
                                          << ibv_wc_status_str(wc.status) << " " << wc.status << " "
                                          << static_cast<int>(wc.wr_id) << " " << wc.vendor_err;
      WorkRequestId* wr_id = reinterpret_cast<WorkRequestId*>(wc.wr_id);
      IBVerbsQP* qp = wr_id->qp;
      switch (wc.opcode) {
        case IBV_WC_RDMA_READ: {
          qp->ReadDone(wr_id);
          break;
        }
        case IBV_WC_SEND: {
          qp->SendDone(wr_id);
          break;
        }
        case IBV_WC_RECV: {
          qp->RecvDone(wr_id);
          break;
        }
        default: UNIMPLEMENTED();
      }
    }
  }
}

const int32_t IBVerbsCommNet::max_poll_wc_num_ = 128;

}  // namespace oneflow

#endif  // WITH_RDMA && PLATFORM_POSIX
