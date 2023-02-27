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
#include "oneflow/core/lazy/actor/actor.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/lazy/stream_context/include/stream_context.h"

namespace oneflow {

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
  CHECK_JUST_MSG(kernel_ctx->stream()->GetAsyncError(), kernel->op_conf().name());
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

}  // namespace

Actor::~Actor() = default;

void Actor::Init(const JobDesc* job_desc, ActorContext* actor_ctx) {
  actor_ctx_ = actor_ctx;
  const TaskProto& task_proto = actor_ctx->task_proto();
  actor_id_ = task_proto.task_id();
  thrd_id_ = ThrdId4ActorId(actor_id_);
  job_id_ = task_proto.job_id();
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel_ctx.reset(new KernelContextImpl(actor_ctx));
    ek.kernel = ConstructKernel(node.kernel_conf(), ek.kernel_ctx.get());
    exec_kernel_vec_.emplace_back(std::move(ek));
  }

  is_kernel_launch_synchronized_ =
      std::all_of(exec_kernel_vec_.cbegin(), exec_kernel_vec_.cend(),
                  [](const ExecKernel& ek) { return ek.kernel->IsKernelLaunchSynchronized(); });
  if (!is_kernel_launch_synchronized_) { CHECK_EQ(exec_kernel_vec_.size(), 1); }

  remaining_eord_cnt_ = 0;
  msg_handler_ = nullptr;
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
  CheckInplaceRegstDescId(task_proto);
  TakeOverInplaceConsumedAndProduced(task_proto.produced_regst_desc());
  is_naive_consumed_eord_ = false;
  TakeOverNaiveConsumed(task_proto.consumed_regst_desc_id());
  TakeOverNaiveProduced(task_proto.produced_regst_desc());
  InitBnInOp2BlobInfo(task_proto);
  VirtualActorInit(task_proto);
}

void Actor::TakeOverInplaceConsumedAndProduced(
    const PbMap<std::string, RegstDescProto>& produced_ids) {
  for (const auto& pair : produced_ids) {
    int64_t out_regst_desc_id = pair.second.regst_desc_id();
    if (pair.second.has_inplace_consumed_regst_desc_id() == false) { continue; }
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

void Actor::TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  auto res = GetNaiveOrCustomizedConsumedRegstDescName();
  bool is_naive_names = res.first == RegstNameType::kNaive;
  const HashSet<std::string>& names = res.second;

  for (const auto& pair : consumed_ids) {
    bool find_the_name = names.find(pair.first) != names.end();
    if (is_naive_names == find_the_name || pair.first == "in_ctrl") {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        if (inplace_consumed_rs_.HasRegstDescId(regst_desc_id)) { continue; }
        naive_consumed_rs_.InsertRegstDescId(regst_desc_id);
      }
    }
  }
  naive_consumed_rs_.InitedDone();
}

void Actor::TakeOverNaiveProduced(const PbMap<std::string, RegstDescProto>& produced_ids) {
  auto res = GetNaiveOrCustomizedProducedRegstDescName();
  bool is_naive_names = res.first == RegstNameType::kNaive;
  const HashSet<std::string>& names = res.second;

  for (const auto& pair : produced_ids) {
    bool find_the_name = names.find(pair.first) != names.end();
    if (inplace_produced_rs_.HasRegstDescId(pair.second.regst_desc_id())) { continue; }
    if (is_naive_names == find_the_name || pair.first.substr(0, 9) == "out_ctrl_") {
      naive_produced_rs_.InsertRegstDescId(pair.second.regst_desc_id());
    }
  }
  naive_produced_rs_.InitedDone();

  for (const auto& pair : produced_regsts_) {
    if (naive_produced_rs_.HasRegstDescId(pair.first) == false) { continue; }
    for (const auto& regst : pair.second) {
      CHECK_EQ(0, naive_produced_rs_.TryPushBackRegst(regst.get()));
    }
  }
}

void Actor::InitBnInOp2BlobInfo(const TaskProto& task_proto) {
  for (int64_t i = 0; i < exec_kernel_vec_.size(); ++i) {
    ExecKernel& ek = exec_kernel_vec_.at(i);
    const ExecNodeProto& node = task_proto.exec_sequence().exec_node(i);
    for (auto& pair : node.kernel_conf().op_attribute().arg_signature().bn_in_op2lbi()) {
      BlobInfo blob_info;
      blob_info.lbi = pair.second;
      const std::string& bn = pair.first;
      auto regst_desc_id_it = node.bn_in_op2regst_desc_id().find(bn);
      if (regst_desc_id_it != node.bn_in_op2regst_desc_id().end()
          && Singleton<RegstMgr>::Get()->HasRegstDescId(regst_desc_id_it->second)) {
        const int64_t regst_desc_id = regst_desc_id_it->second;
        blob_info.regst_desc_id = regst_desc_id;
        const RtRegstDesc& regst_desc =
            Singleton<RegstMgr>::Get()->RegstDesc4RegstDescId(regst_desc_id);
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
      ek.bn_in_op2blob_info.emplace(bn, std::move(blob_info));
    }
  }
}

void Actor::ForEachProducedRegst(const std::function<void(Regst*)>& Handler) const {
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) { Handler(regst.get()); }
  }
}

int64_t Actor::Name2SoleRegstDescId(const std::string& name) const {
  auto find_it = name2regst_desc_id_.find(name);
  if (find_it != name2regst_desc_id_.end()) {
    CHECK_EQ(find_it->second.size(), 1);
    return find_it->second.front();
  }
  return -1;
}

const std::vector<int64_t>& Actor::Name2RegstDescIds(const std::string& name) const {
  return name2regst_desc_id_.at(name);
}

int64_t Actor::ReadingCnt4ProducedRegst(Regst* regst) const {
  return produced_regst2reading_cnt_.at(regst);
}

void Actor::IncreaseReadingCnt4ProducedRegst(Regst* regst, int64_t val) {
  produced_regst2reading_cnt_.at(regst) += val;
}

void Actor::ForEachCurNaiveReadableDataRegst(const std::function<void(const Regst*)>& func) const {
  naive_consumed_rs_.ForEachFrontRegst([func](int64_t regst_desc_id, Regst* regst) {
    if (Singleton<RegstMgr>::Get()->HasProducerTaskId4RegstDescId(regst_desc_id)) { return; }
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) { func(regst); }
  });
}

bool Actor::ReceiveEordMsg(int64_t regst_desc_id) const {
  return eord_regst_desc_ids_.find(regst_desc_id) != eord_regst_desc_ids_.end();
}

int Actor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
    CHECK(eord_regst_desc_ids_.insert(msg.eord_regst_desc_id()).second);
    if (naive_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_naive_consumed_eord_ = true;
    } else if (inplace_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_inplace_consumed_eord_ = true;
    } else {
      NormalProcessCustomizedEordMsg(msg);
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == GlobalProcessCtx::Rank()) {
      Regst* regst = msg.regst();
      if (naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst));
        const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(regst->regst_desc_id());
        CHECK(rdeq.empty() == false);
        if (rdeq.front()->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          NormalProcessNaiveReadableDataRegstMsg(rdeq);
        }
      } else if (inplace_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        CHECK_EQ(0, inplace_consumed_rs_.TryPushBackRegst(regst));
      } else if (TryUpdtStateAsProducedRegst(regst) == 0) {
        // do nothing
      } else {
        NormalProcessCustomizedReadableRegstMsg(msg);
      }
    } else {
      if (NormalTryProcessReadableMsgFromOtherMachine(msg) == false) {
        // process ctrl msg from other rank
        if (IsConsumedCtrlRegstDescId(msg.regst_desc_id())) {
          Regst* regst = msg.regst();
          CHECK(naive_consumed_rs_.HasRegstDescId(msg.regst_desc_id()));
          CHECK(Singleton<RegstMgr>::Get()->HasProducerTaskId4RegstDescId(msg.regst_desc_id()));
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
  // handler halts
  bool has_naive_or_inplace = naive_consumed_rs_.total_regst_desc_cnt() != 0
                              || inplace_consumed_rs_.total_regst_desc_cnt() != 0;
  bool naive_or_inplace_eord_and_empty =
      (is_naive_consumed_eord_ || is_inplace_consumed_eord_)
      && (naive_consumed_rs_.available_regst_desc_cnt() == 0
          && inplace_consumed_rs_.available_regst_desc_cnt() == 0);
  bool customized_eord = IsCustomizedReadAlwaysUnReadyFromNow();
  if ((has_naive_or_inplace && naive_or_inplace_eord_and_empty)
      || (!has_naive_or_inplace && customized_eord)) {
    CHECK_EQ(naive_consumed_rs_.available_regst_desc_cnt(), 0);
    AsyncReturnAllCustomizedReadableRegst();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    if (remaining_eord_cnt_ == 0 && total_reading_cnt_ == 0) {
      OF_SET_MSG_HANDLER(nullptr);
      return 1;
    } else {
      OF_SET_MSG_HANDLER(&Actor::HandlerZombie);
      return 0;
    }
  }
  return 0;
}

int Actor::HandlerZombie(const ActorMsg& msg) {
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

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady()) {
    PrepareProducedNaiveInplaceDataRegst();
    Act();

    AsyncSendCustomizedProducedRegstMsgToConsumer();
    AsyncSendNaiveProducedRegstMsgToConsumer();
    AsyncSendInplaceProducedRegstMsgToConsumer();

    AsyncSendCustomizedConsumedRegstMsgToProducer();
    AsyncSendNaiveConsumedRegstMsgToProducer();
    AsyncRetInplaceConsumedRegstIfNoConsumer();

    AsyncSendQueuedMsg();
  }
  // NOTE(liujuncheng): return inplace consumed
  AsyncSendQueuedMsg();
}

void Actor::AsyncSendNaiveProducedRegstMsgToConsumer() {
  VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  AsyncSendProducedCtrlRegstMsgToConsumer();
}

void Actor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
}

void Actor::AsyncSendInplaceProducedRegstMsgToConsumer() {
  VirtualAsyncSendInplaceProducedRegstMsgToConsumer();
}

void Actor::AsyncRetInplaceConsumedRegstIfNoConsumer() {
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

void Actor::VirtualAsyncSendInplaceProducedRegstMsgToConsumer() {
  HandleProducedInplaceDataRegstToConsumer();
}

void Actor::AsyncSendNaiveConsumedRegstMsgToProducer() {
  VirtualAsyncSendNaiveConsumedRegstMsgToProducer();
  AsyncSendConsumedCtrlRegstMsgToProducer();
}

void Actor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer();
}

void Actor::AsyncSendConsumedCtrlRegstMsgToProducer() {
  auto IsChosenRegstDescId = [this](int64_t regst_desc_id) {
    return IsConsumedCtrlRegstDescId(regst_desc_id) && ConsumedCtrlRegstValid(regst_desc_id);
  };

  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForChosenRegstDeq(IsChosenRegstDescId, [&](int64_t regst_desc_id,
                                                                const std::deque<Regst*>& reg_deq) {
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

void Actor::AsyncSendProducedCtrlRegstMsgToConsumer() {
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

int64_t Actor::HandleRegstToConsumer(Regst* regst) {
  auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK_EQ(regst_reading_cnt_it->second, 0);

  int64_t real_consumer_cnt = 0;
  ActorMsg tpl_msg = ActorMsg::BuildRegstMsgToConsumer(actor_id_, 0, regst);
  for (int64_t consumer : regst->consumers_actor_id()) {
    tpl_msg.set_dst_actor_id(consumer);
    EnqueueAsyncMsg(tpl_msg);
    real_consumer_cnt += 1;
  }
  total_reading_cnt_ += real_consumer_cnt;
  regst_reading_cnt_it->second += real_consumer_cnt;
  return real_consumer_cnt;
}

bool Actor::IsReadReady() const {
  return naive_consumed_rs_.IsCurSlotReady() && inplace_consumed_rs_.IsCurSlotReady()
         && IsCustomizedReadReady();
}

bool Actor::IsWriteReady() const {
  return naive_produced_rs_.IsCurSlotReady() && inplace_produced_rs_.IsCurSlotReady()
         && IsCustomizedWriteReady();
}

void Actor::AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    CHECK_NOTNULL(dynamic_cast<KernelContextImpl*>(ek.kernel_ctx.get()))
        ->UpdateBnInOp2BlobFn([&](const std::string& bn_in_op) -> Blob* {
          const auto blob_info_it = ek.bn_in_op2blob_info.find(bn_in_op);
          if (blob_info_it == ek.bn_in_op2blob_info.cend()) { return nullptr; }
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
    ek.kernel->Launch(ek.kernel_ctx.get());
  }
}

void Actor::AsyncLaunchKernel() {
  AsyncLaunchKernel([](int64_t) -> Regst* {
    UNIMPLEMENTED();
    return nullptr;
  });
}

void Actor::PrepareProducedNaiveInplaceDataRegst() {
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
        CHECK(regst->body_mem_ptr() == nullptr);
        void* body_ptr = nullptr;
        CHECK_JUST(actor_ctx_->stream_ctx()->stream()->AllocAsync(
            &body_ptr, regst->regst_desc()->BodyByteSize4OneRegst()));
        regst->ResetBodyMemPtr(body_ptr);
      } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
        // do nothing
      } else {
        UNIMPLEMENTED();
      }
    }
  });

  inplace_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_data_regst_desc());
    const int64_t in_regst_desc_id = inplace_regst_desc_id_out2in_.at(regst->regst_desc_id());
    Regst* in_regst = inplace_consumed_rs_.Front(in_regst_desc_id);
    CHECK(in_regst != nullptr);
    if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
      CHECK(regst->body_mem_ptr() == nullptr);
      regst->ResetBodyMemPtr(in_regst->body_mem_ptr());
    } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
  });
}

void Actor::HandleProducedNaiveDataRegstToConsumer() {
  tmp_regst_desc_id_vec_.clear();
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      int64_t real_consumer_cnt = HandleRegstToConsumer(regst);
      if (real_consumer_cnt > 0) {
        tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id());
      } else {
        if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
          CHECK_JUST(actor_ctx_->stream_ctx()->stream()->FreeAsync(regst->body_mem_ptr()));
          regst->ResetBodyMemPtr(nullptr);
        } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
          // do nothing
        } else {
          UNIMPLEMENTED();
        }
      }
    }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void Actor::HandleProducedInplaceDataRegstToConsumer() {
  tmp_regst_desc_id_vec_.clear();
  inplace_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_data_regst_desc());
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst);
    if (real_consumer_cnt > 0) {
      tmp_regst_desc_id_vec_.emplace_back(regst->regst_desc_id());
    } else {
      if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
        regst->ResetBodyMemPtr(nullptr);
      } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
        // do nothing
      } else {
        UNIMPLEMENTED();
      }
    }
  });
  inplace_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void Actor::HandleConsumedNaiveDataRegstToProducer() {
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

void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
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

void Actor::AsyncSendRegstMsgToProducer(Regst* regst) {
  AsyncSendRegstMsgToProducer(regst, regst->producer_actor_id());
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst, int64_t producer) {
  // must access regst before sending it to producer
  int64_t regst_desc_id = regst->regst_desc_id();
  EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst));
  naive_consumed_rs_.TryPopFrontRegst(regst_desc_id);
}

Regst* Actor::GetSoleProducedRegst4RegstDescId(int64_t regst_desc_id) const {
  auto it = produced_regsts_.find(regst_desc_id);
  CHECK(it != produced_regsts_.end());
  CHECK_EQ(it->second.size(), 1);
  return it->second.front().get();
}

int Actor::TryUpdtStateAsProducedRegst(Regst* regst) {
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
    if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
      regst->ResetBodyMemPtr(nullptr);
    } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
    AsyncSendRegstMsgToProducer(in_regst);
    CHECK_EQ(0, inplace_consumed_rs_.TryPopFrontRegst(in_regst_desc_id));
  } else if (naive_produced_rs_.TryPushBackRegst(regst) == 0) {
    if (regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
      CHECK_JUST(actor_ctx_->stream_ctx()->stream()->FreeAsync(regst->body_mem_ptr()));
      regst->ResetBodyMemPtr(nullptr);
    } else if (regst->allocation_type() == RegstAllocationType::kStatic) {
      // do nothing
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UpdtStateAsCustomizedProducedRegst(regst);
  }
  return 0;
}

void Actor::EnqueueAsyncMsg(const ActorMsg& msg) {
  if (is_kernel_launch_synchronized_ && thrd_id_ == ThrdId4ActorId(msg.dst_actor_id())) {
    sync_msg_queue_.emplace_back(msg);
  } else {
    async_msg_queue_.emplace_back(msg);
  }
}

Regst* Actor::GetNaiveOrInplaceCurReadable(int64_t regst_desc_id) const {
  Regst* regst = naive_consumed_rs_.Front(regst_desc_id);
  if (regst == nullptr) { regst = inplace_consumed_rs_.Front(regst_desc_id); }
  return regst;
}

Regst* Actor::GetNaiveOrInplaceCurWriteable(int64_t regst_desc_id) const {
  Regst* regst = naive_produced_rs_.Front(regst_desc_id);
  if (regst == nullptr) { regst = inplace_produced_rs_.Front(regst_desc_id); }
  return regst;
}

Regst* Actor::GetNaiveCurReadable(int64_t regst_desc_id) const {
  return naive_consumed_rs_.Front(regst_desc_id);
}

Regst* Actor::GetNaiveCurWriteable(int64_t regst_desc_id) const {
  return naive_produced_rs_.Front(regst_desc_id);
}

void Actor::AsyncSendQueuedMsg() {
  if (!sync_msg_queue_.empty()) {
    Singleton<ActorMsgBus>::Get()->SendMsgsWithoutCommNet(sync_msg_queue_.data(),
                                                          sync_msg_queue_.size(), thrd_id_);
    sync_msg_queue_.clear();
  }
  if (!async_msg_queue_.empty()) {
    std::deque<ActorMsg> msgs;
    msgs.swap(async_msg_queue_);
    AddCallback([msgs]() {
      for (const ActorMsg& msg : msgs) { Singleton<ActorMsgBus>::Get()->SendMsg(msg); }
    });
  }
}

void Actor::AddCallback(std::function<void()> callback) {
  actor_ctx_->AddCallback(std::move(callback));
}

}  // namespace oneflow
