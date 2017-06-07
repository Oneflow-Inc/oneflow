#include "oneflow/core/actor/actor.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

void Actor::Init(const TaskProto& task_proto) {
  // actor_id
  actor_id_ = task_proto.id();
  // ward_func
  if (task_proto.is_forward()) {
    ward_func_ = &Kernel::Forward;
  } else {
    ward_func_ = &Kernel::Backward;
  }
  // exec_kernel_vec_
  exec_kernel_vec_.reserve(task_proto.exec_sequence().exec_node_size());
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = KernelMgr::Singleton().GetKernelFromOpName(node.op_name());
    ek.bn_in_op2regst_desc_id = PbMap2HashMap(node.bn_in_op2regst_desc_id());
    exec_kernel_vec_.push_back(std::move(ek));
  }
  // produced_regst_vec_
  for (const auto& pair : task_proto.produced_regst_desc()) {
    RegstMgr::Singleton().NewRegsts(pair.second, [this](Regst* regst) {
      produced_regst_vec_.emplace_back(regst);
    });
  }
  // name2regst_desc_id_
  for (const auto& pair : task_proto.produced_regst_desc()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second.regst_desc_id()).second);
  }
  for (const auto& pair : task_proto.subscribed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second).second);
  }
  // Status of Produced Registers
  for (const auto& regst : produced_regst_vec_) {
    writeable_produced_regst_[regst->regst_desc_id()].push(regst.get());
    produced_regst2reading_cnt_[regst.get()] = 0;
  }
  writeable_produced_regst_desc_num_ = writeable_produced_regst_.size();
}

void Actor::WardKernel(
    std::function<std::shared_ptr<RegstWarpper>(uint64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    (ek.kernel->*ward_func_)([&](const std::string& bn_in_op) {
      uint64_t regst_desc_id = ek.bn_in_op2regst_desc_id.at(bn_in_op);
      auto regst = Regst4RegstDescId(regst_desc_id);
      const std::string& lbn = ek.kernel->GetLbnFromBnInOp(bn_in_op);
      return regst->GetBlobPtrFromLbn(lbn);
    });
  }
}

void Actor::ForEachProducedRegst(std::function<void(Regst*)> func) {
  for (const auto& regst : produced_regst_vec_) {
    func(regst.get());
  }
}

int Actor::TryOneReadDone(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_regst2reading_cnt_.end()) { return -1; }
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  if (reading_cnt_it->second == 0) {
    auto writeable_produced_regst_it =
        writeable_produced_regst_.find(regst->regst_desc_id());
    if (writeable_produced_regst_it->second.empty()) {
      writeable_produced_regst_desc_num_ += 1;
    }
    writeable_produced_regst_it->second.push(regst);
  }
  return 0;
}

Regst* Actor::GetCurWriteableRegst(uint64_t regst_desc_id) {
  auto it = writeable_produced_regst_.find(regst_desc_id);
  if (it == writeable_produced_regst_.end()) { return nullptr; }
  return it->second.front();
}

void Actor::ForEachCurWriteableRegst(std::function<void(Regst*)> func) {
  for (const auto& pair : writeable_produced_regst_) {
    func(pair.second.front());
  }
}

void Actor::CurWriteDone() {
  for (auto& pair : writeable_produced_regst_) {
    Regst* regst = pair.second.front();
    produced_regst2reading_cnt_.at(regst) = regst->subscribers_actor_id().size();
    for (uint64_t subscriber : regst->subscribers_actor_id()) {
      ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstReader(
            subscriber, regst));
    }
    if (!regst->subscribers_actor_id().empty()) { pair.second.pop(); }
    if (pair.second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
  }
}

bool Actor::IsWriteReady() {
  return writeable_produced_regst_desc_num_ == writeable_produced_regst_.size();
}

} // namespace oneflow
