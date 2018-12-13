#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/nccl_comm_manager.h"
#include "oneflow/core/device/nccl_util.h"
#include "nccl_comm_manager.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace {

std::string GenNcclUniqueIdRpcKey(int64_t group_id) {
  return "nccl_unique_id_" + std::to_string(group_id);
}

int32_t GetDeviceId4Task(int64_t task_id) {
  int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(task_id);
  return static_cast<int32_t>(Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id));
}

int64_t GetMachineId4CommDesc(const NcclCommDesc& comm_desc) {
  return Global<IDMgr>::Get()->MachineId4ActorId(comm_desc.global_device_id());
}

bool IsCommOnThisMachine(const NcclCommDesc& comm_desc) {
  return Global<IDMgr>::Get()->MachineId4ActorId(comm_desc.global_device_id())
         == Global<MachineCtx>::Get()->this_machine_id();
}

void GetOrCreateNcclUniqueId(const std::vector<const NcclCommDesc*>& comm_descs, int64_t group_id,
                             int32_t num_rank, ncclUniqueId* unique_id) {
  if (comm_descs.front()->rank_id() == 0) {
    NcclCheck(ncclGetUniqueId(unique_id));
    if (comm_descs.size() < num_rank) {
      Global<CtrlClient>::Get()->PushKV(GenNcclUniqueIdRpcKey(group_id),
                                        std::string(unique_id->internal, NCCL_UNIQUE_ID_BYTES));
    }
  } else {
    Global<CtrlClient>::Get()->PullKV(
        GenNcclUniqueIdRpcKey(group_id), [&unique_id](const std::string& val) {
          memcpy(unique_id->internal, val.data(), NCCL_UNIQUE_ID_BYTES);
        });
  }
}

void CreateNcclComms4CommDescs(const std::vector<const NcclCommDesc*>& comm_descs,
                               const ncclUniqueId& unique_id, int32_t num_rank,
                               std::vector<ncclComm_t>* comms) {
  comms->resize(comm_descs.size());
  NcclCheck(ncclGroupStart());
  FOR_RANGE(size_t, i, 0, comm_descs.size()) {
    const NcclCommDesc* comm_desc = comm_descs.at(i);
    const int32_t device_id = GetDeviceId4Task(comm_desc->global_device_id());
    cudaSetDevice(device_id);
    NcclCheck(ncclCommInitRank(&comms->at(i), num_rank, unique_id,
                               static_cast<int32_t>(comm_desc->rank_id())));
  }
  NcclCheck(ncclGroupEnd());
}

}  // namespace

NcclCommMgr::NcclCommMgr(const Plan& plan) {
  const NcclTopo& topo = plan.nccl_topo();
  for (const NcclCommGroup& group : topo.group()) {
    std::vector<const NcclCommDesc*> local_comm_descs;
    for (const NcclCommDesc& comm_desc : group.comm_desc()) {
      if (IsCommOnThisMachine(comm_desc)) { local_comm_descs.push_back(&comm_desc); }
    }
    if (local_comm_descs.empty()) { continue; }
    ncclUniqueId nccl_unique_id{};
    GetOrCreateNcclUniqueId(local_comm_descs, group.id(), group.comm_desc_size(), &nccl_unique_id);
    std::vector<ncclComm_t> comms_vec;
    CreateNcclComms4CommDescs(local_comm_descs, nccl_unique_id, group.comm_desc_size(), &comms_vec);
    FOR_RANGE(size_t, i, 0, local_comm_descs.size()) {
      comm_desc_id2comm_.emplace(local_comm_descs.at(i)->id(), comms_vec[i]);
    }
  }
  for (const auto& pair : topo.task_id2comm_desc_id()) {
    actor_id2comm_desc_id_[pair.first] = pair.second;
  }
}

NcclCommMgr::~NcclCommMgr() {
  for (const auto& pair : comm_desc_id2comm_) { ncclCommDestroy(pair.second); }
}

ncclComm_t NcclCommMgr::NcclComm4ActorId(int64_t actor_id) const {
  auto comm_id_it = actor_id2comm_desc_id_.find(actor_id);
  if (comm_id_it == actor_id2comm_desc_id_.end()) { return nullptr; }
  auto comm_it = comm_desc_id2comm_.find(comm_id_it->second);
  if (comm_it == comm_desc_id2comm_.end()) { return nullptr; }
  return comm_it->second;
}

}  // namespace oneflow

#endif  // WITH_CUDA
