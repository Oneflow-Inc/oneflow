/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/lazy/actor/of_collective_actor.h"
#include <cstdint>
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/lazy/actor/actor_message.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/lazy/stream_context/include/stream_context.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

using namespace boxing::of_collective;

namespace {

class KernelContextImpl : public KernelContext, public ActorContextProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelContextImpl);
  explicit KernelContextImpl(ActorContext* actor_ctx)
      : actor_ctx_(actor_ctx),
        stream_ctx_(actor_ctx->stream_ctx()),
        state_(nullptr),
        stream_kernel_observer_(nullptr) {
    auto* kernel_observer_provider = dynamic_cast<KernelObserverProvider*>(stream_ctx_);
    if (kernel_observer_provider != nullptr) {
      stream_kernel_observer_ = kernel_observer_provider->GetKernelObserver();
    }
  }
  ~KernelContextImpl() = default;

  ep::Stream* stream() const override { return stream_ctx_->stream(); }

  ActorContext* GetActorContext() const override { return actor_ctx_; }

  Blob* BnInOp2Blob(const std::string& bn) const override { return bn_in_op2blob_fn_(bn); }

  const std::shared_ptr<KernelState>& state() const override { return state_; }

  void set_state(std::shared_ptr<KernelState> state) override { state_ = std::move(state); }

  void WillForward(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForward(KernelContext* kernel_ctx, const Kernel* kernel) override;

  void WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override;

  void WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override;
  void DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override;

  void UpdateBnInOp2BlobFn(std::function<Blob*(const std::string&)> fn) {
    bn_in_op2blob_fn_ = std::move(fn);
  }

 private:
  ActorContext* actor_ctx_;
  StreamContext* stream_ctx_;
  std::function<Blob*(const std::string&)> bn_in_op2blob_fn_;
  std::shared_ptr<KernelState> state_;
  KernelObserver* stream_kernel_observer_;
};

void KernelContextImpl::WillForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->WillForward(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForward(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->DidForward(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->DidForward(kernel_ctx, kernel);
  }
}

void KernelContextImpl::WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->WillForwardHeader(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForwardHeader(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->DidForwardHeader(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->DidForwardHeader(kernel_ctx, kernel);
  }
}

void KernelContextImpl::WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->WillForwardDataContent(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForwardDataContent(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  Singleton<KernelObserver>::Get()->DidForwardDataContent(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->DidForwardDataContent(kernel_ctx, kernel);
  }
}

void CheckInplaceRegstDescId(const TaskProto& task_proto) {
  HashSet<int64_t> consumed_regst_desc_ids;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (int64_t id : pair.second.regst_desc_id()) { consumed_regst_desc_ids.insert(id); }
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.has_inplace_consumed_regst_desc_id() == false) { continue; }
    int64_t in_regst_desc_id = pair.second.inplace_consumed_regst_desc_id();
    CHECK(consumed_regst_desc_ids.find(in_regst_desc_id) != consumed_regst_desc_ids.end());
  }
}

void show_cached_nego_ready_msg(const HashMap<int64_t, oneflow::ActorMsg>& cached_nego_ready_msg, int64_t actor_id, HashMap<ActorMsgType, std::string>& print_actor_msg_type, HashMap<CollectiveNegoCmd, std::string>& print_nego_cmd) {
  VLOG(2) << "========= show begin ======== " << actor_id;
  if (cached_nego_ready_msg.empty()) {
    VLOG(2) << "Actor " << actor_id << " cached_nego_ready_msg is empty";
  } else {
    FOR_EACH(src_actor_id7msg, cached_nego_ready_msg) {
      VLOG(2) << "Actor " << actor_id << " in the cached_nego_ready_msg, src: " << src_actor_id7msg->first << " dst: " << src_actor_id7msg->second.dst_actor_id() << " msg.type:  " << print_actor_msg_type[src_actor_id7msg->second.msg_type()] << " collective_nego_cmd: " << print_nego_cmd[src_actor_id7msg->second.collective_nego_cmd()];
    }
  }
  VLOG(2) << "========= show end ======== " << actor_id;
}

}  // namespace

void OfCollectiveActor::Init(const JobDesc* job_desc, ActorContext* actor_ctx) {
  actor_ctx_ = actor_ctx;
  const TaskProto& task_proto = actor_ctx->task_proto();
  actor_id_ = task_proto.task_id();
  thrd_id_ = ThrdId4ActorId(actor_id_);
  job_id_ = task_proto.job_id();
  CHECK_EQ(task_proto.exec_sequence().exec_node_size(), 1);
  const ExecNodeProto& node = actor_ctx->task_proto().exec_sequence().exec_node()[0];
  op_name_ = node.kernel_conf().op_attribute().op_conf().name();

  const OfRequestId& request_id = Singleton<CollectiveMgr>::Get()->GetOfRequestIdByName(op_name_);
  auto* token = Singleton<CollectiveMgr>::Get()->CreateOfRequestEntryToken(request_id);
  auto* request_entry = Singleton<CollectiveMgr>::Get()->GetOfRequestEntry(token);
  Singleton<CollectiveMgr>::Get()->DestroyOfRequestEntryToken(token);
  nego_tree_info_ = request_entry->nego_tree_topo()[actor_id_];
  is_nego_root_ = (nego_tree_info_.upstream_id == -1);
  is_nego_leaf_ = (nego_tree_info_.downstream_id.size() == 0);
  // if (nego_tree_info_.downstream_id.size() > 0) {
  //   for (int i = 0; i < nego_tree_info_.downstream_id.size(); ++i)
  // } else {
  // }
  print_nego_cmd_.emplace(CollectiveNegoCmd::kCollectiveReady, "kCollectiveReady");
  print_nego_cmd_.emplace(CollectiveNegoCmd::kCollectiveStart, "kCollectiveStart");
  print_nego_cmd_.emplace(CollectiveNegoCmd::kCollectiveDone, "kCollectiveDone");
  
  print_status_.emplace(CollectiveStatus::kInvalid, "kInvalid");
  print_status_.emplace(CollectiveStatus::kLocalReady, "kLocalReady");
  print_status_.emplace(CollectiveStatus::kDownstreamReady, "kDownstreamReady");
  print_status_.emplace(CollectiveStatus::kCanAct, "kCanAct");

  print_actor_msg_type_.emplace(ActorMsgType::kCmdMsg, "kCmdMsg");
  print_actor_msg_type_.emplace(ActorMsgType::kCollectiveMsg, "kCollectiveMsg");
  print_actor_msg_type_.emplace(ActorMsgType::kEordMsg, "kEordMsg");
  print_actor_msg_type_.emplace(ActorMsgType::kRegstMsg, "kRegstMsg");

  if (!is_nego_leaf_) {
    cached_nego_ready_msg_ = HashMap<int64_t, ActorMsg>();
    ready_downstream_id_ = HashSet<int64_t>();
  }
  cached_nego_ready_msg_ = HashMap<int64_t, ActorMsg>();
  ready_downstream_id_ = HashSet<int64_t>();
  ResetCollectiveStatus();

  ek_.kernel_ctx.reset(new KernelContextImpl(actor_ctx));
  ek_.kernel = ConstructKernel(node.kernel_conf(), ek_.kernel_ctx.get());

  is_kernel_launch_synchronized_ = ek_.kernel->IsKernelLaunchSynchronized();

  remaining_eord_cnt_ = 0;
  eord_regst_desc_ids_.clear();

  for (const auto& pair : task_proto.produced_regst_desc()) {
    Singleton<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
    if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
      produced_ctrl_regst_desc_ids_.insert(regst_desc_id);
    }
  }
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) { produced_regst2reading_cnt_[regst.get()] = 0; }
  }

  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.find(pair.first) == name2regst_desc_id_.end());
    std::vector<int64_t>& regst_desc_id_vec = name2regst_desc_id_[pair.first];
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      regst_desc_id_vec.emplace_back(regst_desc_id);
    }
    remaining_eord_cnt_ += pair.second.regst_desc_id_size();
    if (pair.first == "in_ctrl") {
      consumed_ctrl_regst_desc_ids_.insert(regst_desc_id_vec.begin(), regst_desc_id_vec.end());
    }
  }

  total_reading_cnt_ = 0;

  is_inplace_consumed_eord_ = false;
  is_naive_consumed_eord_ = false;

  CheckInplaceRegstDescId(task_proto);
  TakeOverInplaceConsumedAndProduced(task_proto.produced_regst_desc());
  TakeOverNaiveConsumed(task_proto.consumed_regst_desc_id());
  TakeOverNaiveProduced(task_proto.produced_regst_desc());
  InitBnInOp2BlobInfo(task_proto);

  OF_SET_MSG_HANDLER(&OfCollectiveActor::HandlerNormal);
}

void OfCollectiveActor::ResetCollectiveStatus() {
  received_downstream_ready_cnt_ = 0;
  collective_status_ = CollectiveStatus::kInvalid;
  VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];
  ready_downstream_id_.clear();
  // should not clear cached_nego_ready_msg_ here.
}

void OfCollectiveActor::TakeOverInplaceConsumedAndProduced(
    const PbMap<std::string, RegstDescProto>& produced_ids) {
  for (const auto& pair : produced_ids) {
    if (pair.second.has_inplace_consumed_regst_desc_id() == false) { continue; }

    int64_t out_regst_desc_id = pair.second.regst_desc_id();
    int64_t in_regst_desc_id = pair.second.inplace_consumed_regst_desc_id();
    inplace_regst_desc_id_in2out_.insert(std::make_pair(in_regst_desc_id, out_regst_desc_id));
    inplace_regst_desc_id_out2in_.insert(std::make_pair(out_regst_desc_id, in_regst_desc_id));
    inplace_consumed_rs_.InsertRegstDescId(in_regst_desc_id);
    inplace_produced_rs_.InsertRegstDescId(out_regst_desc_id);
  }
  inplace_consumed_rs_.InitedDone();
  inplace_produced_rs_.InitedDone();
  for (const auto& pair : produced_regsts_) {
    if (inplace_produced_rs_.HasRegstDescId(pair.first)) {
      for (const auto& regst : pair.second) {
        CHECK_EQ(0, inplace_produced_rs_.TryPushBackRegst(regst.get()));
        if (regst->consumers_actor_id().size() == 0) {
          CHECK(inplace_in_ids_with_no_out_consumed_
                    .emplace(inplace_regst_desc_id_out2in_.at(pair.first))
                    .second);
        }
      }
    }
  }
}

void OfCollectiveActor::TakeOverNaiveConsumed(
    const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  for (const auto& pair : consumed_ids) {
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      if (inplace_consumed_rs_.HasRegstDescId(regst_desc_id)) { continue; }
      naive_consumed_rs_.InsertRegstDescId(regst_desc_id);
    }
  }
  naive_consumed_rs_.InitedDone();
}

void OfCollectiveActor::TakeOverNaiveProduced(
    const PbMap<std::string, RegstDescProto>& produced_ids) {
  for (const auto& pair : produced_ids) {
    if (inplace_produced_rs_.HasRegstDescId(pair.second.regst_desc_id())) { continue; }
    naive_produced_rs_.InsertRegstDescId(pair.second.regst_desc_id());
  }
  naive_produced_rs_.InitedDone();

  for (const auto& pair : produced_regsts_) {
    if (naive_produced_rs_.HasRegstDescId(pair.first) == false) { continue; }
    for (const auto& regst : pair.second) {
      CHECK_EQ(0, naive_produced_rs_.TryPushBackRegst(regst.get()));
    }
  }
}

void OfCollectiveActor::InitBnInOp2BlobInfo(const TaskProto& task_proto) {
  const ExecNodeProto& node = task_proto.exec_sequence().exec_node(0);
  for (auto& pair : node.kernel_conf().op_attribute().arg_signature().bn_in_op2lbi()) {
    BlobInfo blob_info;
    const std::string& bn = pair.first;
    blob_info.lbi = pair.second;
    // Map<string, int>
    auto regst_desc_id_it = node.bn_in_op2regst_desc_id().find(bn);
    if (regst_desc_id_it != node.bn_in_op2regst_desc_id().end()
        && Singleton<RegstMgr>::Get()->HasRegstDescId(regst_desc_id_it->second)) {
      const int64_t regst_desc_id = regst_desc_id_it->second;
      blob_info.regst_desc_id = regst_desc_id;
      const RtRegstDesc& regst_desc = Singleton<RegstMgr>::Get()->RegstDesc4RegstDescId(regst_desc_id);
      blob_info.ordinal = regst_desc.GetOrdinalForLbi(blob_info.lbi);
      if (naive_produced_rs_.HasRegstDescId(regst_desc_id)) {
        blob_info.rs = &naive_produced_rs_;
      } else if (inplace_produced_rs_.HasRegstDescId(regst_desc_id)) {
        blob_info.rs = &inplace_produced_rs_;
      } else if (naive_consumed_rs_.HasRegstDescId(regst_desc_id)) {
        blob_info.rs = &naive_consumed_rs_;
      } else if (inplace_consumed_rs_.HasRegstDescId(regst_desc_id)) {
        blob_info.rs = &inplace_consumed_rs_;
      } else {
        blob_info.rs = nullptr;
      }
    } else {
      blob_info.regst_desc_id = -1;
      blob_info.ordinal = -1;
      blob_info.rs = nullptr;
    }
    ek_.bn_in_op2blob_info.emplace(bn, std::move(blob_info));
  }
}

int64_t OfCollectiveActor::ReadingCnt4ProducedRegst(Regst* regst) const {
  return produced_regst2reading_cnt_.at(regst);
}

void OfCollectiveActor::IncreaseReadingCnt4ProducedRegst(Regst* regst, int64_t val) {
  produced_regst2reading_cnt_.at(regst) += val;
}

int OfCollectiveActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
    CHECK(eord_regst_desc_ids_.insert(msg.eord_regst_desc_id()).second);
    if (naive_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_naive_consumed_eord_ = true;
    } else if (inplace_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_inplace_consumed_eord_ = true;
    } else {
      UNIMPLEMENTED();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (is_nego_root_) {
      VLOG(2) << "NEGOROOT GET Actor " << actor_id_ << " status: " << print_status_[collective_status_] << " get kRegstMsg from " << msg.src_actor_id();
    }
    if (msg.SrcMachineId() == GlobalProcessCtx::Rank()) {
      if (is_nego_root_) {
        VLOG(2) << "NEGOROOT Actor " << actor_id_ << " msg.SrcMachineId() == GlobalProcessCtx::Rank()";
      }
      Regst* regst = msg.regst();
      if (naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        if (is_nego_root_) {
          VLOG(2) << "NEGOROOT Actor " << actor_id_ << " naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())";
        }
        CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst));
        const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(regst->regst_desc_id());
        CHECK(rdeq.empty() == false);

        if (rdeq.front()->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          // TODO: (Panlichen) currently do nothing, maybe useless for us.
          NormalProcessNaiveReadableDataRegstMsg(rdeq);
        }

      } else if (inplace_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        if (is_nego_root_) {
          VLOG(2) << "NEGOROOT Actor " << actor_id_ << " inplace_consumed_rs_.HasRegstDescId(regst->regst_desc_id()))";
        }
        CHECK_EQ(0, inplace_consumed_rs_.TryPushBackRegst(regst));
        int64_t out_regst_desc_id = inplace_regst_desc_id_in2out_.at(regst->regst_desc_id());
        CHECK(regst->GetSoleBlob()->dptr()
              == inplace_produced_rs_.Front(out_regst_desc_id)->GetSoleBlob()->dptr());
      } else if (TryUpdtStateAsProducedRegst(regst) == 0) {
        if (is_nego_root_) {
          VLOG(2) << "NEGOROOT Actor " << actor_id_ << " TryUpdtStateAsProducedRegst(regst) == 0";
        }
        // do nothing
      } else {
        UNIMPLEMENTED();
      }
    } else {
      if (is_nego_root_) {
        VLOG(2) << "NEGOROOT Actor " << actor_id_ << " from other GlobalProcessCtx::Rank()";
      }
      // can only process ctrl msg from other processes
      if (NormalTryProcessReadableMsgFromOtherMachine(msg) == false) {
        // process ctrl msg from other rank
        if (IsConsumedCtrlRegstDescId(msg.regst_desc_id())) {
          if (is_nego_root_) {
            VLOG(2) << "NEGOROOT Actor " << actor_id_ << " IsConsumedCtrlRegstDescId(msg.regst_desc_id())";
          }
          Regst* regst = msg.regst();
          CHECK(naive_consumed_rs_.HasRegstDescId(msg.regst_desc_id()));
          CHECK(Singleton<RegstMgr>::Get()->HasProducerTaskId4RegstDescId(msg.regst_desc_id()));
          CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst, msg.regst_desc_id()));
          const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(msg.regst_desc_id());
          CHECK(rdeq.empty() == false);
        } else {
          if (is_nego_root_) {
            VLOG(2) << "NEGOROOT Actor " << actor_id_ << " NOT IsConsumedCtrlRegstDescId(msg.regst_desc_id())";
          }
          CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
        }
      }
    }
    // for debug, skip negotiation and run normal regst-actor stuff directly
    if (ParseBooleanFromEnv("ONEFLOW_OFCCL_SKIP_NEGO", false)) {
      collective_status_ = CollectiveStatus::kCanAct;
      VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];
    } else {
      if (IsReadReady() && IsWriteReady()) {
        if (is_nego_root_) {
          VLOG(2) << "NEGOROOT Actor " << actor_id_ << " Checking IsReadReady() && IsWriteReady()";
        }
        // every actor gets here.
        collective_status_ = CollectiveStatus::kLocalReady;
        VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];

        // leaf updates to kDownstreamReady directly.
        if (is_nego_leaf_) {
          collective_status_ = CollectiveStatus::kDownstreamReady;
          VLOG(2) << "Actor " << actor_id_ << " Leaf directly UpdateCollectiveStatus to " << print_status_[collective_status_];

          if (!is_nego_root_) {
            // send ready msg to upstream.
            auto ready_msg = 
              ActorMsg::BuildCollectiveMsg(actor_id_, nego_tree_info_.upstream_id,
                                            CollectiveNegoCmd::kCollectiveReady);
            VLOG(2) << "SEND Actor " << actor_id_  << " leave send kCollectiveReady to " << nego_tree_info_.upstream_id;
            SyncSendMsg(std::move(ready_msg));
          }
        } else {
          // non-leaf need to check whether there is cached ds_ready msg
          VLOG(2) << "Actor " << actor_id_  << " ready to process cached_nego_ready_msg_, then show_cached_nego_ready_msg: cached_nego_ready_msg_.size() is " << cached_nego_ready_msg_.size();
          // show_cached_nego_ready_msg(cached_nego_ready_msg_, actor_id_, print_actor_msg_type_, print_nego_cmd_);
          
          int64_t cached_nego_ready_msg_id = 0;
          FOR_EACH(src_actor_id7msg, cached_nego_ready_msg_) {
            VLOG(2) << "Actor " << actor_id_ << " invoke ReactToNegoCmd for cached_nego_ready_msg_ " << cached_nego_ready_msg_id << " out of " << cached_nego_ready_msg_.size() << " msg.src: " << src_actor_id7msg->second.src_actor_id() << " msg.dst: " << src_actor_id7msg->second.dst_actor_id() << " msg.type: " << print_actor_msg_type_[src_actor_id7msg->second.msg_type()];
            ++cached_nego_ready_msg_id;

            ReactToNegoCmd(src_actor_id7msg->second);
          }

          cached_nego_ready_msg_.clear();

          VLOG(2) << "Actor " << actor_id_  << " clear cached_nego_ready_msg_, then show_cached_nego_ready_msg: cached_nego_ready_msg_.size() is " << cached_nego_ready_msg_.size();
          // show_cached_nego_ready_msg(cached_nego_ready_msg_, actor_id_, print_actor_msg_type_, print_nego_cmd_);
        }
      }
    }
  } else if (msg.msg_type() == ActorMsgType::kCollectiveMsg) {
    VLOG(2) << "GET Actor " << actor_id_ << " status: " << print_status_[collective_status_] << " get " << print_nego_cmd_[msg.collective_nego_cmd()] << " from " << msg.src_actor_id();
    VLOG(2) << "Actor " << actor_id_  << " invoke ReactToNegoCmd after recv kCollectiveMsg from " << msg.src_actor_id();

    ReactToNegoCmd(msg);
  } else {
    UNIMPLEMENTED();
  }
  // all consumed regsts get eord
  bool naive_or_inplace_gets_eord_and_both_empty =
      (is_naive_consumed_eord_ || is_inplace_consumed_eord_)
      && (naive_consumed_rs_.available_regst_desc_cnt() == 0
          && inplace_consumed_rs_.available_regst_desc_cnt() == 0);
  if (naive_or_inplace_gets_eord_and_both_empty) {
    CHECK_EQ(naive_consumed_rs_.available_regst_desc_cnt(), 0);
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (remaining_eord_cnt_ == 0 && total_reading_cnt_ == 0) {
      OF_SET_MSG_HANDLER(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLER(&OfCollectiveActor::HandlerZombie);
      return 0;
    }
  }
  return 0;
}

int OfCollectiveActor::HandlerZombie(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    CHECK_GE(remaining_eord_cnt_, 1);
    remaining_eord_cnt_ -= 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) { AsyncSendRegstMsgToProducer(msg.regst()); }
  } else {
    UNIMPLEMENTED();
  }
  if (remaining_eord_cnt_ == 0 && total_reading_cnt_ == 0) {
    msg_handler_ = nullptr;
    return 1;
  }
  return 0;
}

void OfCollectiveActor::ReactToNegoCmd(const ActorMsg& msg) {
  VLOG(2) << "Actor " << actor_id_  << " ReactToNegoCmd get msg of type " << print_actor_msg_type_[msg.msg_type()];
  CHECK(msg.msg_type() == ActorMsgType::kCollectiveMsg);
  int64_t src_actor_id = msg.src_actor_id();

  switch (msg.collective_nego_cmd()) {
    case CollectiveNegoCmd::kCollectiveReady:
      if (IsInDebugMode()) {
        CHECK(std::find(nego_tree_info_.downstream_id.begin(), nego_tree_info_.downstream_id.end(), src_actor_id) != nego_tree_info_.downstream_id.end());
      }
      // every downstream should only send one ready msg.
      CHECK(ready_downstream_id_.find(src_actor_id) == ready_downstream_id_.end()) << " Actor " << actor_id_;
      switch (collective_status_) {
        case CollectiveStatus::kInvalid:
          cached_nego_ready_msg_.emplace(src_actor_id, msg);

          VLOG(2) << "Actor " << actor_id_  << " get kCollectiveReady when kInvalid from " << src_actor_id << ", then show_cached_nego_ready_msg: cached_nego_ready_msg_.size() is " << cached_nego_ready_msg_.size();
          // show_cached_nego_ready_msg(cached_nego_ready_msg_, actor_id_, print_actor_msg_type_, print_nego_cmd_);
        break;

        case CollectiveStatus::kLocalReady:

          ++received_downstream_ready_cnt_;
          ready_downstream_id_.emplace(src_actor_id);
          if (IsDownstreamReady()) {
            collective_status_ = CollectiveStatus::kDownstreamReady;
            VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];
            if (!is_nego_root_) {
              // send ready msg to upstream.
              auto ready_msg = 
                ActorMsg::BuildCollectiveMsg(actor_id_, nego_tree_info_.upstream_id,
                                            CollectiveNegoCmd::kCollectiveReady);
              VLOG(2) << "SEND Actor " << actor_id_  << " middle send kCollectiveReady to " << nego_tree_info_.upstream_id;
              SyncSendMsg(std::move(ready_msg));
            } else {
              collective_status_ = CollectiveStatus::kCanAct;
              VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];
            }
          }
          break;
        default: 
          // CollectiveStatus::kDownstreamReady, CollectiveStatus::kCanAct
          LOG(FATAL) << "Actor " << actor_id_ << " In status " << print_status_[collective_status_] << " should not get " << "kCollectiveReady";
          break;
      }
    break;

    case CollectiveNegoCmd::kCollectiveStart:
      CHECK(collective_status_ == CollectiveStatus::kDownstreamReady) << "Actor " << actor_id_ << " In status " << print_status_[collective_status_] << " should not get " << "kCollectiveStart";
      CHECK(src_actor_id == nego_tree_info_.upstream_id) << "Actor " << actor_id_;

      collective_status_ = CollectiveStatus::kCanAct;
      VLOG(2) << "Actor " << actor_id_ << " UpdateCollectiveStatus to " << print_status_[collective_status_];

    break;

    case CollectiveNegoCmd::kCollectiveDone:
      CHECK(collective_status_ == CollectiveStatus::kCanAct) << "Actor " << actor_id_ << " In status " << print_status_[collective_status_] << " should not get " << "kCollectiveStart";

      VLOG(1) << "Actor " << actor_id_ << " receive kCollectiveDone and will return regsts and ResetCollectiveStatus";
      
      AsyncSendNaiveProducedRegstMsgToConsumer();
      // no inplace ctrl regst
      HandleProducedInplaceDataRegstToConsumer();

      AsyncSendNaiveConsumedRegstMsgToProducer();
      // inplace regst with further consumer is handled in HandlerNormal
      AsyncRetInplaceConsumedRegstIfNoConsumer();

      AsyncSendQueuedMsg();

      ResetCollectiveStatus();

    break;

    default:
      UNIMPLEMENTED();
      break;
  }

  if (CanAct()) {
    if (!is_nego_leaf_) {
      FOR_EACH(downstream_id, nego_tree_info_.downstream_id) {
        auto start_msg =
          ActorMsg::BuildCollectiveMsg(actor_id_, *downstream_id,
                                        CollectiveNegoCmd::kCollectiveStart);
        VLOG(2) << "SEND Actor " << actor_id_  << " Send kCollectiveStart to " << *downstream_id;
        SyncSendMsg(std::move(start_msg));
      }
    }
    if (is_nego_root_) {
      VLOG(2) << "NEGOROOT Actor " << actor_id_  << " goes to Act() in ReactToNegoCmd";
    }
    Act();
    if (is_nego_root_) {
      VLOG(2) << "NEGOROOT Actor " << actor_id_  << " return from Act() in ReactToNegoCmd";
    }
  }
}

bool OfCollectiveActor::IsReadReady() const {
  return naive_consumed_rs_.IsCurSlotReady() && inplace_consumed_rs_.IsCurSlotReady();
}

bool OfCollectiveActor::IsWriteReady() const {
  return naive_produced_rs_.IsCurSlotReady() && inplace_produced_rs_.IsCurSlotReady();
}

bool OfCollectiveActor::IsDownstreamReady() const {
  // can also handle NO DOWNSTREAM, where downstream_id.size() is 0
  return received_downstream_ready_cnt_ == nego_tree_info_.downstream_id.size();
}

void OfCollectiveActor::AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId) {
  VLOG(2) << "Actor " << actor_id_ << " Enter OfCollectiveActor::AsyncLaunchKernel";
  CHECK_NOTNULL(dynamic_cast<KernelContextImpl*>(ek_.kernel_ctx.get()))
    ->UpdateBnInOp2BlobFn([&](const std::string& bn_in_op) -> Blob* {
      const auto blob_info_it = ek_.bn_in_op2blob_info.find(bn_in_op);
      if (blob_info_it == ek_.bn_in_op2blob_info.cend()) { return nullptr; }
      const BlobInfo& info = blob_info_it->second;
      if (info.regst_desc_id == -1) { return nullptr; }
      Regst* regst = nullptr;
      if (info.rs != nullptr) {
        regst = info.rs->Front(info.regst_desc_id);
      } else {
        regst = Regst4RegstDescId(info.regst_desc_id);
      }
      if (regst == nullptr) { return nullptr; }
      if (info.ordinal >= 0) {
        return regst->GetBlobByOrdinal(info.ordinal);
      } else {
        return regst->GetBlobByLbi(info.lbi);
      }
    });
  ek_.kernel->Launch(ek_.kernel_ctx.get());
  VLOG(2) << "Actor " << actor_id_ << " OfCollectiveActor::AsyncLaunchKernel Done";
}

void OfCollectiveActor::Act() {
  VLOG(2) << "Actor " << actor_id_ << " Enter OfCollectiveActor::Act()";
  
  CHECK(IsReadReady() && IsWriteReady() && CanAct()) << "Actor " << actor_id_;
  
  AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* { return nullptr; });
  int64_t actor_id = actor_id_;
  AddCallback([actor_id](){
    Singleton<ActorMsgBus>::Get()->SendMsg(
      ActorMsg::BuildCollectiveMsg(actor_id, actor_id, CollectiveNegoCmd::kCollectiveDone)
    );
  });

  VLOG(1) << "Actor " << actor_id_ << " will send itself kCollectiveDone after kernel finish";

  VLOG(2) << "Actor " << actor_id_ << " OfCollectiveActor::Act() Done";
  return;
}

void OfCollectiveActor::SyncSendMsg(const ActorMsg& msg) {
  Singleton<ActorMsgBus>::Get()->SendMsg(msg);
}

void OfCollectiveActor::AsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
  AsyncSendProducedCtrlRegstMsgToConsumer();
}

void OfCollectiveActor::HandleProducedNaiveDataRegstToConsumer() {
  tmp_regst_desc_id_vec_.clear();
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      int64_t real_consumer_cnt = HandleRegstToConsumer(regst);
      if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id()); }
    }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

int64_t OfCollectiveActor::HandleRegstToConsumer(Regst* regst) {
  auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK_EQ(regst_reading_cnt_it->second, 0);

  int64_t real_consumer_cnt = 0;
  for (int64_t consumer : regst->consumers_actor_id()) {
    EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
    real_consumer_cnt += 1;
  }
  total_reading_cnt_ += real_consumer_cnt;
  regst_reading_cnt_it->second += real_consumer_cnt;
  return real_consumer_cnt;
}

void OfCollectiveActor::AsyncSendProducedCtrlRegstMsgToConsumer() {
  auto IsChosenRegstDescId = [this](int64_t regst_desc_id) {
    return IsProducedCtrlRegstDescId(regst_desc_id) && ProducedCtrlRegstValid(regst_desc_id);
  };
  tmp_regst_desc_id_vec_.clear();
  naive_produced_rs_.ForChosenFrontRegst(IsChosenRegstDescId, [&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_ctrl_regst_desc());
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst);
    if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id()); }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void OfCollectiveActor::HandleProducedInplaceDataRegstToConsumer() {
  tmp_regst_desc_id_vec_.clear();
  inplace_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_data_regst_desc());
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst);
    if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id()); }
  });
  inplace_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void OfCollectiveActor::AsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer();
  AsyncSendConsumedCtrlRegstMsgToProducer();
}

void OfCollectiveActor::HandleConsumedNaiveDataRegstToProducer() {
  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForEachFrontRegst([&](int64_t regst_desc_id, Regst* regst) {
    if (IsConsumedCtrlRegstDescId(regst_desc_id)) { return; }
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      // must access regst before sending it to producer
      tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id());
      EnqueueAsyncMsg(
          ActorMsg::BuildRegstMsgToProducer(actor_id_, regst->producer_actor_id(), regst));
    }
  });
  naive_consumed_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void OfCollectiveActor::AsyncSendConsumedCtrlRegstMsgToProducer() {
  auto IsChosenRegstDescId = [this](int64_t regst_desc_id) {
    return IsConsumedCtrlRegstDescId(regst_desc_id) && ConsumedCtrlRegstValid(regst_desc_id);
  };

  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForChosenRegstDeq(
      IsChosenRegstDescId, [&](int64_t regst_desc_id, const std::deque<Regst*>& reg_deq) {
        CHECK(reg_deq.empty() == false);
        auto producer_task_id = Singleton<RegstMgr>::Get()->ProducerTaskId4RegstDescId(regst_desc_id);
        Regst* regst = reg_deq.front();
        CHECK_GE(reg_deq.size(), 1);
        // must access regst before sending it to producer
        tmp_regst_desc_id_vec_.emplace_back(regst_desc_id);
        EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, producer_task_id, regst));
      });
  naive_consumed_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void OfCollectiveActor::AsyncRetInplaceConsumedRegstIfNoConsumer() {
  tmp_regst_desc_id_vec_.clear();
  inplace_consumed_rs_.ForChosenRegstDeq(
      [&](int64_t regst_desc_id) {
        return inplace_in_ids_with_no_out_consumed_.find(regst_desc_id)
               != inplace_in_ids_with_no_out_consumed_.end();
      },
      [&](const std::deque<Regst*>& deq) {
        if (!deq.empty()) {
          Regst* in_regst = deq.front();
          CHECK(in_regst);
          AsyncSendRegstMsgToProducer(in_regst);
          tmp_regst_desc_id_vec_.emplace_back(in_regst->regst_desc_id());
        }
      });
  inplace_consumed_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void OfCollectiveActor::EnqueueAsyncMsg(const ActorMsg& msg) {
  if (is_kernel_launch_synchronized_ && thrd_id_ == ThrdId4ActorId(msg.dst_actor_id())) {
    Singleton<ActorMsgBus>::Get()->SendMsg(msg);
  } else {
    async_msg_queue_.emplace_back(msg);
  }
}

void OfCollectiveActor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (auto& pair : produced_regsts_) {
    CHECK(!pair.second.empty());
    const RtRegstDesc* regst_desc = pair.second.front()->regst_desc();
    AddCallback([regst_desc]() {
      for (int64_t consumer : regst_desc->consumers_actor_id()) {
        Singleton<ActorMsgBus>::Get()->SendMsg(
            ActorMsg::BuildEordMsg(consumer, regst_desc->regst_desc_id()));
      }
    });
  }
}

void OfCollectiveActor::AsyncSendRegstMsgToProducer(Regst* regst) {
  AsyncSendRegstMsgToProducer(regst, regst->producer_actor_id());
}

void OfCollectiveActor::AsyncSendRegstMsgToProducer(Regst* regst, int64_t producer) {
  // must access regst before sending it to producer
  int64_t regst_desc_id = regst->regst_desc_id();
  EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst));
  // only naive needs to pop here.
  naive_consumed_rs_.TryPopFrontRegst(regst_desc_id);
}

int OfCollectiveActor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_regst2reading_cnt_.end()) { return -1; }
  CHECK(produced_regsts_.find(regst->regst_desc_id()) != produced_regsts_.end());
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_cnt_ -= 1;
  if (reading_cnt_it->second != 0) { return 0; }

  if (inplace_produced_rs_.TryPushBackRegst(regst) == 0) {
    int64_t in_regst_desc_id = inplace_regst_desc_id_out2in_.at(regst->regst_desc_id());
    Regst* in_regst = inplace_consumed_rs_.Front(in_regst_desc_id);
    CHECK(in_regst);
    AsyncSendRegstMsgToProducer(in_regst);
    CHECK_EQ(0, inplace_consumed_rs_.TryPopFrontRegst(in_regst_desc_id));
  } else if (naive_produced_rs_.TryPushBackRegst(regst) != 0) {
    UNIMPLEMENTED();
  }
  return 0;
}

void OfCollectiveActor::AsyncSendQueuedMsg() {
  if (!async_msg_queue_.empty()) {
    std::deque<ActorMsg> msgs;
    msgs.swap(async_msg_queue_);
    AddCallback([msgs]() {
      for (const ActorMsg& msg : msgs) { Singleton<ActorMsgBus>::Get()->SendMsg(msg); }
    });
  }
}

void OfCollectiveActor::AddCallback(std::function<void()> callback) {
  actor_ctx_->AddCallback(std::move(callback));
}

REGISTER_ACTOR(TaskType::kOfCollectiveBoxingGeneric, OfCollectiveActor);
}  // namespace oneflow
