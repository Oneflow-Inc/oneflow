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

void Actor::ProcessEord() {
  VLOG(4) << "actor " << actor_id_ << " process one eord";
  num_of_remaining_eord_ -= 1;
  if (!num_of_remaining_eord_) {
    if (num_of_read_empty_) {
      if (!total_reading_cnt_) {
        OF_SET_MSG_HANDLER(nullptr);
      } else {
        OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
      }
      AsyncSendEORDMsgForAllProducedRegstDesc();
    } else {
      OF_SET_MSG_HANDLER(&Actor::HandlerWaitUntilNoReadableRegst);
    }
  }
}

void Actor::TrySwitchToZombie() {
  if (total_reading_cnt_ == 0) {
    OF_SET_MSG_HANDLER(nullptr);
  } else {
    OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
  }
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

int Actor::HandlerZombie(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    return 0;
  }
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  if (total_reading_cnt_ == 0) {
    msg_handler_ = nullptr;
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
    (ek.kernel->*launch_func_)(
        kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
          auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
          if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) {
            return nullptr;
          }
          auto regst = Regst4RegstDescId(regst_desc_id_it->second);
          const std::string& lbn = ek.kernel->Lbn4BnInOp(bn_in_op);
          return regst->GetBlobPtrFromLbn(lbn);
        });
  }
  VLOG(4) << "actor " << actor_id_ << " launch kernel for piece_id "
          << expected_piece_id_;
  expected_piece_id_ += 1;
}

void Actor::AsyncSendReadableRegstMsg(
    std::function<void(Regst*)> RegstPreProcess,
    std::function<bool(int64_t)> IsAllowedActor) {
  for (auto& pair : writeable_produced_regst_) {
    Regst* regst = pair.second.front();
    RegstPreProcess(regst);
    auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
    CHECK_EQ(regst_reading_cnt_it->second, 0);
    for (int64_t subscriber : regst->subscribers_actor_id()) {
      if (!IsAllowedActor(subscriber)) { continue; }
      total_reading_cnt_ += 1;
      regst_reading_cnt_it->second += 1;
      device_ctx_->AddCallBack([subscriber, regst]() {
        ActorMsg msg = ActorMsg::BuildReadableRegstMsg(subscriber, regst);
        ActorMsgBus::Singleton()->SendMsg(std::move(msg));
      });
    }
    if (!regst->subscribers_actor_id().empty()) { pair.second.pop(); }
    if (pair.second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
    VLOG(4) << "actor " << actor_id() << " "
            << "send readable register " << regst << ", "
            << "regst_desc_id:" << regst->regst_desc_id() << ", "
            << "this_regst_reading_cnt:" << regst_reading_cnt_it->second << ", "
            << "total_reading_cnt:" << total_reading_cnt_;
  }
}

void Actor::AsyncSendReadableRegstMsg(
    std::function<void(Regst*)> RegstPreProcess) {
  AsyncSendReadableRegstMsg(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::AsyncSendReadableRegstMsg(
    std::function<bool(int64_t)> IsAllowedActor) {
  AsyncSendReadableRegstMsg([](Regst*) {}, IsAllowedActor);
}

void Actor::AsyncSendReadableRegstMsg() {
  AsyncSendReadableRegstMsg([](Regst*) {});
}

void Actor::AsyncSendEORDMsgToSubscribers(int64_t regst_desc_id) {
  VLOG(4) << "actor " << actor_id_ << " "
          << "send eord for regst_desc_id:" << regst_desc_id;
  const RtRegstDesc* regst_desc =
      produced_regsts_.at(regst_desc_id).front()->regst_desc();
  device_ctx_->AddCallBack([regst_desc]() {
    for (int64_t subscriber : regst_desc->subscribers_actor_id()) {
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
  VLOG(4) << "actor " << actor_id_ << " "
          << "return register " << wp->regst_raw_ptr() << " "
          << "to actor " << wp->producer_actor_id() << ", "
          << "regst_desc_id:" << wp->regst_desc_id();
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
  VLOG(4) << "actor " << actor_id() << "\'s "
          << "reading_cnt for " << regst << " -= 1, "
          << "current cnt:" << reading_cnt_it->second << ", "
          << "current total_reading_cnt:" << total_reading_cnt_ << ", "
          << "regst_desc_id: " << regst->regst_desc_id();
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
