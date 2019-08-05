#include "oneflow/core/actor/op_actor_context.h"

namespace oneflow {

namespace actor {

void OpActorCtx::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  actor_id_ = task_proto.task_id();
  act_id_ = -1;
  remaining_eord_cnt_ = 0;
  total_reading_cnt_ = 0;
  eord_regst_desc_ids_.clear();
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

  {
    for (const auto& pair : task_proto.produced_regst_desc()) {
      Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
        produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
      });
      int64_t regst_desc_id = pair.second.regst_desc_id();
      CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
      produced_regst2expected_act_id_[regst_desc_id] = act_id_;
    }
    for (const auto& pair : produced_regsts_) {
      for (const auto& regst : pair.second) { produced_regst2reading_cnt_[regst.get()] = 0; }
    }
  }

  {
    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      CHECK(name2regst_desc_id_.find(pair.first) == name2regst_desc_id_.end());
      std::vector<int64_t>& regst_desc_id_vec = name2regst_desc_id_[pair.first];
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        regst_desc_id_vec.push_back(regst_desc_id);
      }
      remaining_eord_cnt_ += pair.second.regst_desc_id_size();
    }
  }

  rs_wrappers_.emplace_back(RSWrapperType::kCtrl, new CtrlRSWrapper);
  VirtualExpandRegstSlotWrapper();
  for (auto& pair : rs_wrappers_) { pair.second->Init(task_proto); }

  VirtualSetMsgHandler();
  VirtualInitCustomized(task_proto);
}

void CtrlRSWrapper::DerivedInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
      InsertNewRegstDescId(true, pair.second.regst_desc_id());
    }
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.first == "in_ctrl") {
      for (int regst_desc_id : pair.second.regst_desc_id()) {
        InsertNewRegstDescId(false, pair.second.regst_desc_id());
      }
    }
  }
}

void NaiveRSWrapper::DerivedInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_naive_regst_desc()) {
      InsertNewRegstDescId(true, pair.second.regst_desc_id());
    }
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_naive_regst_desc()) {
      InsertNewRegstDescId(false, pair.second.regst_desc_id());
    }
  }
}

}

}
