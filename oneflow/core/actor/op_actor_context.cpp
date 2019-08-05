#include "oneflow/core/actor/op_actor_context.h"
#include "oneflow/core/actor/regst_pattern_wrapper.h"

namespace oneflow {

namespace actor {


void OpActorCtx::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  actor_id_ = task_proto.task_id();
  act_id_ = -1;
  InitDeviceCtx(thread_ctx);
  if (task_proto.has_parallel_ctx()) {
    parallel_ctx_.reset(new ParallelContext(task_proto.parallel_ctx()));
  }
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = ConstructKernel(parallel_ctx(), node.kernel_conf(), device_ctx_.get());
    ek.bn_in_op2regst_desc_id = PbMap2HashMap(node.bn_in_op2regst_desc_id());
    exec_kernel_vec_.push_back(std::move(ek));
  }

  for (const auto& pair : task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
    produced_regst2expected_act_id_[regst_desc_id] = act_id_;
  }

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.find(pair.first) == name2regst_desc_id_.end());
    std::vector<int64_t>& regst_desc_id_vec = name2regst_desc_id_[pair.first];
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      regst_desc_id_vec.push_back(regst_desc_id);
    }
  }

  InsertRegstPattern(new CtrlPatternWrapper);
  VirtualHandleRegstPattern(task_proto);
  for (auto& pair : wrappers_) {
    pair.second->Init(task_proto, produced_regsts_);
    pair.second->ForEachRegstDescId([&](int64_t regst_desc_id) {
          CHECK(regst_desc_id2wrapper_.emplace(regst_desc_id, pair.second.get()));
    });
  }

  VirtualSetMsgHandler();
}

void OpActorCtx::UpdateWithRegstMsg(const ActorMsg& msg) {
  regst_desc_id2wrapper_.at(msg.regst()->regst_desc_id())->UpdateWithRegstMsg(msg);
}

void OpActorCtx::UpdateWithEordMsg(const ActorMsg& msg) {
  regst_desc_id2wrapper_.at(msg.regst()->regst_desc_id())->UpdateWithEordMsg(msg);
}

void OpActorCtx::UpdateWithCmdMsg(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
}

void OpActorCtx::IsReady4Act() const {
  for (const auto& pair : wrappers_) {
    if (pair.second->IsReady4Act() == false) {
      return false;
    }
  }
  return true;
}

void OpActorCtx::Act() {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    //TODO(niuchong): BnInOp2Blob return nullptr or failed?
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      int64_t regst_desc_id = ek.bn_in_op2regst_desc_id.at(bn_in_op);
      RegstPatternWrapperIf* wrapper = regst_desc_id2wrapper_.at(regst_desc_id);
      Regst* regst = wrapper->GetRegstByRegstDescId(regst_desc_id);
      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
      return regst->GetBlobByLbi(lbi);
    });
  }
}

void OpActorCtx::HandleRegstMsgAfterAct() {
  for (auto& pair : wrappers_) {
    pair.second->HandleRegstMsgAfterAct();
  }
}

void OpActorCtx::NoLongerConsumeRegst() const {
  for (const auto& pair : wrappers_) {
    if (pair.second->NoLongerConsumeRegst() == false) {
      return false;
    }
  }
  return true;
}

MsgHandler OpActorCtx::initial_msg_handler() const {
  return initial_msg_handler_;
}

void OpActorCtx::InsertRegstPattern(RegstPatternWrapperIf* wrapper) {
  CHECK(wrappers_.emplace(wrapper->type(), wrapper).second);
}

class GeneralOpActorCtx final : public OpActorCtx {
 public:
  GeneralOpActorCtx() = default;
  ~GeneralOpActorCtx() = default;

 private:
  void VirtualHandleRegstPattern(const TaskProto& task_proto) override {
    InsertRegstPattern(new NaivePatternWrapper);
  }
  void VirtualSetMsgHandler() override {
    SetInitMsgHandler(&HandlerUtil::HandlerNormal);
  }
};

}

}
