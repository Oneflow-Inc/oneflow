#include "oneflow/core/actor/actor.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

void Actor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  // actor_id
  actor_id_ = task_proto.id();
  // ward_func
  if (task_proto.is_forward()) {
    launch_func_ = &Kernel::Forward;
  } else {
    launch_func_ = &Kernel::Backward;
  }
  // exec_kernel_vec_
  exec_kernel_vec_.reserve(task_proto.exec_sequence().exec_node_size());
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = KernelMgr::Singleton()->GetKernelFromOpName(node.op_name());
    ek.bn_in_op2regst_desc_id = PbMap2HashMap(node.bn_in_op2regst_desc_id());
    exec_kernel_vec_.push_back(std::move(ek));
  }
  // produced_regsts_
  for (const auto& pair : task_proto.produced_regst_desc()) {
    RegstMgr::Singleton()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
  }
  // name2regst_desc_id_
  for (const auto& pair : task_proto.produced_regst_desc()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second.regst_desc_id())
              .second);
  }
  for (const auto& pair : task_proto.subscribed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second).second);
  }
  // Status of Produced Registers
  expected_piece_id_ = 0;
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) {
      writeable_produced_regst_[regst->regst_desc_id()].push(regst.get());
      produced_regst2reading_cnt_[regst.get()] = 0;
    }
  }
  writeable_produced_regst_desc_num_ = writeable_produced_regst_.size();
  total_reading_cnt_ = 0;
}

int64_t Actor::RegstDescId4Name(const std::string& name) const {
  auto find_it = name2regst_desc_id_.find(name);
  if (find_it != name2regst_desc_id_.end()) { return find_it->second; }
  return -1;
}

KernelCtx Actor::GenDefaultKernelCtx() const {
  KernelCtx ctx;
  ctx.device_ctx = device_ctx_.get();
  return ctx;
}

int Actor::HandleWaitUntilReadingCntEqualZero(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  if (total_reading_cnt_ == 0) {
    msg_handle_ = nullptr;
    return 1;
  }
  return 0;
}

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady()) { Act(); }
}

void Actor::AsyncLaunchKernel(
    const KernelCtx& kernel_ctx,
    std::function<std::shared_ptr<RegstWrapper>(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    (ek.kernel->*launch_func_)(kernel_ctx, [&](const std::string& bn_in_op) {
      int64_t regst_desc_id = ek.bn_in_op2regst_desc_id.at(bn_in_op);
      auto regst = Regst4RegstDescId(regst_desc_id);
      const std::string& lbn = ek.kernel->Lbn4BnInOp(bn_in_op);
      return regst->GetBlobPtrFromLbn(lbn);
    });
  }
  expected_piece_id_ += 1;
}

void Actor::AsyncSendReadableRegstMsg() {
  AsyncSendReadableRegstMsg([](Regst*) {});
}

void Actor::AsyncSendReadableRegstMsg(std::function<void(Regst*)> PreProcess) {
  for (auto& pair : writeable_produced_regst_) {
    Regst* regst = pair.second.front();
    PreProcess(regst);
    device_ctx_->AddCallBack([regst]() {
      for (int64_t subscriber : regst->subscribers_actor_id()) {
        ActorMsg msg = ActorMsg::BuildReadableRegstMsg(subscriber, regst);
        ActorMsgBus::Singleton()->SendMsg(std::move(msg));
      }
    });
    produced_regst2reading_cnt_.at(regst) =
        regst->subscribers_actor_id().size();
    total_reading_cnt_ += regst->subscribers_actor_id().size();
    if (!regst->subscribers_actor_id().empty()) { pair.second.pop(); }
    if (pair.second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
  }
}

void Actor::AsyncSendEORDMsgToSubscribers(int64_t regst_desc_id) {
  Regst* one_regst = produced_regsts_.at(regst_desc_id).front().get();
  device_ctx_->AddCallBack([one_regst]() {
    for (int64_t subscriber : one_regst->subscribers_actor_id()) {
      ActorMsg msg;
      msg.set_dst_actor_id(subscriber);
      msg.set_actor_cmd(ActorCmd::kEORD);
      ActorMsgBus::Singleton()->SendMsg(std::move(msg));
    }
  });
}

void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (const auto& pair : produced_regsts_) {
    AsyncSendEORDMsgToSubscribers(pair.first);
  }
}

void Actor::AsyncDo(std::function<void()> func) {
  device_ctx_->AddCallBack(func);
}

void Actor::AsyncSendRegstMsgToProducer(
    const std::shared_ptr<RegstWrapper>& wp) {
  ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(wp->producer_actor_id(),
                                                   wp->regst_raw_ptr());
  AsyncDo([msg]() { ActorMsgBus::Singleton()->SendMsg(msg); });
}

int Actor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_regst2reading_cnt_.end()) { return -1; }
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_cnt_ -= 1;
  if (reading_cnt_it->second != 0) { return 0; }
  auto writeable_it = writeable_produced_regst_.find(regst->regst_desc_id());
  if (writeable_it == writeable_produced_regst_.end()) { return 0; }
  if (writeable_it->second.empty()) { writeable_produced_regst_desc_num_ += 1; }
  writeable_it->second.push(regst);
  return 0;
}

Regst* Actor::GetCurWriteableRegst(int64_t regst_desc_id) {
  auto it = writeable_produced_regst_.find(regst_desc_id);
  if (it == writeable_produced_regst_.end()) { return nullptr; }
  return it->second.front();
}

Regst* Actor::GetCurWriteableRegst(const std::string& name) {
  return GetCurWriteableRegst(RegstDescId4Name(name));
}

void Actor::ForEachCurWriteableRegst(std::function<void(Regst*)> func) {
  for (const auto& pair : writeable_produced_regst_) {
    func(pair.second.front());
  }
}

bool Actor::IsWriteReady() {
  return writeable_produced_regst_desc_num_ == writeable_produced_regst_.size();
}

void Actor::SetReadOnlyForRegstDescId(int64_t regst_desc_id) {
  auto it = writeable_produced_regst_.find(regst_desc_id);
  if (!it->second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
  writeable_produced_regst_.erase(it);
}

}  // namespace oneflow
