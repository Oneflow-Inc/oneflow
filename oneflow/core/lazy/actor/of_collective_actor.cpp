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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/stream/include/stream_context.h"

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
  Global<KernelObserver>::Get()->WillForward(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForward(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForward(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForward(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->DidForward(kernel_ctx, kernel);
  }
}

void KernelContextImpl::WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->WillForwardHeader(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForwardHeader(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForwardHeader(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->DidForwardHeader(kernel_ctx, kernel);
  }
}

void KernelContextImpl::WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->WillForwardDataContent(kernel_ctx, kernel);
  if (stream_kernel_observer_ != nullptr) {
    stream_kernel_observer_->WillForwardDataContent(kernel_ctx, kernel);
  }
}

void KernelContextImpl::DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) {
  Global<KernelObserver>::Get()->DidForwardDataContent(kernel_ctx, kernel);
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

}  // namespace

// void OfCollectiveActor::Act() {
//   AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* { return nullptr; });
// }

void OfCollectiveActor::Init(const JobDesc* job_desc, ActorContext* actor_ctx) {
  actor_ctx_ = actor_ctx;
  const TaskProto& task_proto = actor_ctx->task_proto();
  actor_id_ = task_proto.task_id();
  thrd_id_ = ThrdId4ActorId(actor_id_);
  job_id_ = task_proto.job_id();
  CHECK_EQ(task_proto.exec_sequence().exec_node_size(), 1);
  const ExecNodeProto& node = actor_ctx->task_proto().exec_sequence().exec_node()[0];
  op_name_ = node.kernel_conf().op_attribute_ref();

  const OfRequestId& request_id = 
    Global<CollectiveMgr>::Get()->GetOfRequestIdByName(op_name_);
  auto* token = Global<CollectiveMgr>::Get()->CreateOfRequestEntryToken(request_id);
  auto* request_entry = Global<CollectiveMgr>::Get()->GetOfRequestEntry(token);
  Global<CollectiveMgr>::Get()->DestroyOfRequestEntryToken(token);
  nego_tree_info_ = std::move(request_entry->nego_tree_topo()[actor_id_]);

  ek_.kernel_ctx.reset(new KernelContextImpl(actor_ctx));
  ek_.kernel = ConstructKernel(node.kernel_conf(), ek_.kernel_ctx.get());

  is_kernel_launch_synchronized_ = ek_.kernel->IsKernelLaunchSynchronized();

  remaining_eord_cnt_ = 0;
  eord_regst_desc_ids_.clear();

  for (const auto& pair : task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
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

void OfCollectiveActor::TakeOverInplaceConsumedAndProduced(const PbMap<std::string, RegstDescProto>& produced_ids) {
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

void OfCollectiveActor::TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  for (const auto& pair : consumed_ids) {
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      if (inplace_consumed_rs_.HasRegstDescId(regst_desc_id)) { continue; }
      naive_consumed_rs_.InsertRegstDescId(regst_desc_id);
    }
  }
  naive_consumed_rs_.InitedDone();
}

void OfCollectiveActor::TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids) {
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
        && Global<RegstMgr>::Get()->HasRegstDescId(regst_desc_id_it->second)) {
      const int64_t regst_desc_id = regst_desc_id_it->second;
      blob_info.regst_desc_id = regst_desc_id;
      const RtRegstDesc& regst_desc =
          Global<RegstMgr>::Get()->RegstDesc4RegstDescId(regst_desc_id);
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
    if (msg.SrcMachineId() == GlobalProcessCtx::Rank()) {
      Regst* regst = msg.regst();
      if (naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst));
        const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(regst->regst_desc_id());
        CHECK(rdeq.empty() == false);

        if (rdeq.front()->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          // TODO: (Panlichen) currently do nothing, maybe useless for us.
          NormalProcessNaiveReadableDataRegstMsg(rdeq);
        }

      } else if (inplace_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        CHECK_EQ(0, inplace_consumed_rs_.TryPushBackRegst(regst));
        int64_t out_regst_desc_id = inplace_regst_desc_id_in2out_.at(regst->regst_desc_id());
        CHECK(regst->GetSoleBlob()->dptr()
                == inplace_produced_rs_.Front(out_regst_desc_id)->GetSoleBlob()->dptr());
      } else if (TryUpdtStateAsProducedRegst(regst) == 0) {
        // do nothing
      } else {
        UNIMPLEMENTED();
      }
    } else  {
      // can only process ctrl msg from other processes
      if (NormalTryProcessReadableMsgFromOtherMachine(msg) == false) {
        // process ctrl msg from other rank
        if (IsConsumedCtrlRegstDescId(msg.regst_desc_id())) {
          Regst* regst = msg.regst();
          CHECK(naive_consumed_rs_.HasRegstDescId(msg.regst_desc_id()));
          CHECK(Global<RegstMgr>::Get()->HasProducerTaskId4RegstDescId(msg.regst_desc_id()));
          CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst, msg.regst_desc_id()));
          const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(msg.regst_desc_id());
          CHECK(rdeq.empty() == false);
        } else {
          CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
        }
      }            
    }
    ActUntilFail();
  } else if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
    ActUntilFail();
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

void OfCollectiveActor::ActUntilFail() {

}

void OfCollectiveActor::EnqueueAsyncMsg(const ActorMsg& msg) {
  if (is_kernel_launch_synchronized_ && thrd_id_ == ThrdId4ActorId(msg.dst_actor_id())) {
    Global<ActorMsgBus>::Get()->SendMsg(msg);
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
        Global<ActorMsgBus>::Get()->SendMsg(
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
      for (const ActorMsg& msg : msgs) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
    });
  }
}

void OfCollectiveActor::AddCallback(std::function<void()> callback) {
  actor_ctx_->AddCallback(std::move(callback));
}

REGISTER_ACTOR(TaskType::kOfCollectiveBoxingGeneric, OfCollectiveActor);
}  // namespace oneflow
