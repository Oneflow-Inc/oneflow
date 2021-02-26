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
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/runtime_job_descs.h"

namespace oneflow {

namespace {

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

void Actor::Init(const JobDesc* job_desc, const TaskProto& task_proto,
                 const ThreadCtx& thread_ctx) {
  job_desc_ = job_desc;
  actor_id_ = task_proto.task_id();
  act_id_ = -1;
  InitDeviceCtx(thread_ctx);
  if (task_proto.has_parallel_ctx()) {
    parallel_ctx_.reset(new ParallelContext(task_proto.parallel_ctx()));
  }
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = ConstructKernel(job_desc_, node.kernel_conf(), device_ctx_.get());
    exec_kernel_vec_.push_back(std::move(ek));
  }

  is_kernel_launch_synchronized_ =
      std::all_of(exec_kernel_vec_.cbegin(), exec_kernel_vec_.cend(),
                  [](const ExecKernel& ek) { return ek.kernel->IsKernelLaunchSynchronized(); });
  if (!is_kernel_launch_synchronized_) { CHECK_EQ(exec_kernel_vec_.size(), 1); }

  remaining_eord_cnt_ = 0;
  msg_handler_ = nullptr;
  eord_regst_desc_ids_.clear();

  for (const auto& pair : task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
    produced_regst2expected_act_id_[regst_desc_id] = act_id_;
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
      regst_desc_id_vec.push_back(regst_desc_id);
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
      ek.bn_in_op2blob_info.emplace(bn, std::move(blob_info));
    }
  }
}

void Actor::ForEachProducedRegst(const std::function<void(Regst*)>& Handler) const {
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) { Handler(regst.get()); }
  }
}

DeviceType Actor::GetDeviceType() const {
  return Global<IDMgr>::Get()->GetDeviceTypeFromActorId(actor_id_);
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

int64_t Actor::GetPieceId4NaiveCurReadableDataRegst() const {
  int64_t init_val = -2;
  int64_t pid = init_val;
  auto FirstFoundOnly = [&pid, init_val](int64_t) { return pid == init_val; };
  naive_consumed_rs_.ForChosenFrontRegst(FirstFoundOnly, [&pid](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) { pid = regst->piece_id(); }
  });
  CHECK_GE(pid, 0);
  return pid;
}

int64_t Actor::GetPieceId4NaiveOrInplaceCurReadableDataRegst() const {
  int64_t init_val = -2;
  int64_t pid = init_val;
  auto FirstFoundOnly = [&pid, init_val](int64_t) { return pid == init_val; };
  auto Select = [&pid](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) { pid = regst->piece_id(); }
  };
  naive_consumed_rs_.ForChosenFrontRegst(FirstFoundOnly, Select);
  if (pid == init_val) { inplace_consumed_rs_.ForChosenFrontRegst(FirstFoundOnly, Select); }
  CHECK_GE(pid, 0);
  return pid;
}

void Actor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  switch (GetDeviceType()) {
    case DeviceType::kCPU: {
      device_ctx_.reset(new CpuDeviceCtx());
      break;
    }
#ifdef WITH_CUDA
    case DeviceType::kGPU: {
      CudaStreamHandle* cuda_handle = nullptr;
      cuda_handle = thread_ctx.g_cuda_stream.get();
      device_ctx_.reset(new CudaDeviceCtx(cuda_handle));
      break;
    }
#endif
    default: { UNIMPLEMENTED(); }
  }
}

KernelCtx Actor::GenDefaultKernelCtx() const {
  KernelCtx ctx;
  ctx.device_ctx = device_ctx_.get();
  return ctx;
}

void Actor::SetReadableRegstInfo(const Regst* regst, ReadableRegstInfo* info) const {
  info->set_regst_desc_id(regst->regst_desc_id());
  info->set_act_id(regst->act_id());
}

void Actor::ForEachCurNaiveReadableDataRegst(std::function<void(const Regst*)> func) const {
  naive_consumed_rs_.ForEachFrontRegst([func](Regst* regst) {
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
        int64_t out_regst_desc_id = inplace_regst_desc_id_in2out_.at(regst->regst_desc_id());
        CHECK(regst->packed_blob()->dptr()
              == inplace_produced_rs_.Front(out_regst_desc_id)->packed_blob()->dptr());
      } else if (TryUpdtStateAsProducedRegst(regst) == 0) {
        // do nothing
      } else {
        NormalProcessCustomizedReadableRegstMsg(msg);
      }
    } else {
      if (NormalTryProcessReadableMsgFromOtherMachine(msg) == false) {
        CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
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

void Actor::TryLogActEvent(const std::function<void()>& DoAct) const {
  if (Global<RuntimeCtx>::Get()->is_experiment_phase() || NeedCollectActEvent()) {
    auto act_event = std::make_shared<ActEvent>();
    act_event->set_is_experiment_phase(Global<RuntimeCtx>::Get()->is_experiment_phase());
    act_event->set_actor_id(actor_id());
    act_event->set_work_stream_id(GetGlobalWorkStreamId());
    act_event->set_act_id(act_id_);
    act_event->set_ready_time(GetCurTime());
    naive_consumed_rs_.ForEachFrontRegst([&](const Regst* readable_regst) {
      ReadableRegstInfo* info = act_event->add_readable_regst_infos();
      Actor::SetReadableRegstInfo(readable_regst, info);
    });
    ForEachCurCustomizedReadableRegst([&](const Regst* readable_regst) {
      ReadableRegstInfo* info = act_event->add_readable_regst_infos();
      SetReadableRegstInfo(readable_regst, info);
    });
    device_ctx_->AddCallBack([act_event]() { act_event->set_start_time(GetCurTime()); });

    DoAct();

    device_ctx_->AddCallBack([act_event]() {
      act_event->set_stop_time(GetCurTime());
      // The stream poller thread is not allowed to perform blocking RPC call. Hence, the
      // RPC call is forwarded to the thread pool and will be executed there.
      Global<ThreadPool>::Get()->AddWork(
          [act_event]() { Global<CtrlClient>::Get()->PushActEvent(*act_event); });
    });
  } else {
    DoAct();
  }
}

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady()) {
    act_id_ += 1;
    TryLogActEvent([&] { Act(); });

    AsyncSendCustomizedProducedRegstMsgToConsumer();
    AsyncSendNaiveProducedRegstMsgToConsumer();
    AsyncSendInplaceProducedRegstMsgToConsumer();

    AsyncSendCustomizedConsumedRegstMsgToProducer();
    AsyncSendNaiveConsumedRegstMsgToProducer();
    AsyncRetInplaceConsumedRegstIfNoConsumer();

    AsyncSendQueuedMsg();
  }
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
          tmp_regst_desc_id_vec_.push_back(in_regst->regst_desc_id());
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
  HandleConsumedNaiveDataRegstToProducer([](Regst* regst) { return true; });
}

void Actor::AsyncSendConsumedCtrlRegstMsgToProducer() {
  auto IsChosenRegstDescId = [this](int64_t regst_desc_id) {
    return IsConsumedCtrlRegstDescId(regst_desc_id) && ConsumedCtrlRegstValid(regst_desc_id);
  };

  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForChosenRegstDeq(IsChosenRegstDescId, [&](const std::deque<Regst*>& reg_deq) {
    CHECK(reg_deq.empty() == false);
    Regst* regst = reg_deq.front();
    CHECK(regst->regst_desc()->regst_desc_type().has_ctrl_regst_desc());
    int32_t returned_regst_num =
        regst->regst_desc()->regst_desc_type().ctrl_regst_desc().returned_regst_num();
    CHECK_GE(returned_regst_num, 1);
    CHECK_GE(reg_deq.size(), returned_regst_num);
    for (size_t i = 0; i < returned_regst_num; ++i) {
      Regst* regst = reg_deq.at(i);
      // must access regst before sending it to producer
      tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id());
      EnqueueAsyncMsg(
          ActorMsg::BuildRegstMsgToProducer(actor_id_, regst->producer_actor_id(), regst));
    }
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
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst, [](int64_t) { return true; });
    if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id()); }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

int64_t Actor::HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor) {
  auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK_EQ(regst_reading_cnt_it->second, 0);
  regst->set_act_id(act_id_);

  int64_t real_consumer_cnt = 0;
  for (int64_t consumer : regst->consumers_actor_id()) {
    if (!IsAllowedActor(consumer)) { continue; }
    EnqueueAsyncMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
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

void Actor::AsyncLaunchKernel(const KernelCtx& kernel_ctx,
                              std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      const auto blob_info_it = ek.bn_in_op2blob_info.find(bn_in_op);
      if (blob_info_it == ek.bn_in_op2blob_info.cend()) { return nullptr; }
      const BlobInfo& info = blob_info_it->second;
      if (info.regst_desc_id == -1) { return nullptr; }
      Regst* regst;
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
  }
}

void Actor::AsyncLaunchKernel(const KernelCtx& kernel_ctx) {
  AsyncLaunchKernel(kernel_ctx, [](int64_t) -> Regst* {
    UNIMPLEMENTED();
    return nullptr;
  });
}

void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                                   std::function<bool(int64_t)> IsAllowedActor) {
  tmp_regst_desc_id_vec_.clear();
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (RegstPreProcess(regst) == false) { return; }
      int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
      if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id()); }
    }
  });
  naive_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess) {
  HandleProducedNaiveDataRegstToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(int64_t)> IsAllowedActor) {
  HandleProducedNaiveDataRegstToConsumer([](Regst*) { return true; }, IsAllowedActor);
}

void Actor::HandleProducedNaiveDataRegstToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([](Regst*) { return true; });
}

void Actor::HandleProducedInplaceDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                                     std::function<bool(int64_t)> IsAllowedActor) {
  tmp_regst_desc_id_vec_.clear();
  inplace_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_data_regst_desc());
    if (RegstPreProcess(regst) == false) { return; }
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
    if (real_consumer_cnt > 0) { tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id()); }
  });
  inplace_produced_rs_.PopFrontRegsts(tmp_regst_desc_id_vec_);
}

void Actor::HandleProducedInplaceDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess) {
  HandleProducedInplaceDataRegstToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::HandleProducedInplaceDataRegstToConsumer(std::function<bool(int64_t)> IsAllowedActor) {
  HandleProducedInplaceDataRegstToConsumer([](Regst*) { return true; }, IsAllowedActor);
}

void Actor::HandleProducedInplaceDataRegstToConsumer() {
  HandleProducedInplaceDataRegstToConsumer([](Regst*) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(Regst* regst) {
  AsyncSendRegstMsgToConsumer(regst, [](int64_t) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor) {
  int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
  if (real_consumer_cnt > 0) { naive_produced_rs_.TryPopFrontRegst(regst->regst_desc_id()); }
}

void Actor::HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst) {
  tmp_regst_desc_id_vec_.clear();
  naive_consumed_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (IsAllowedRegst(regst) == false) { return; }
      // must access regst before sending it to producer
      tmp_regst_desc_id_vec_.push_back(regst->regst_desc_id());
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
    device_ctx_->AddCallBack([regst_desc]() {
      for (int64_t consumer : regst_desc->consumers_actor_id()) {
        Global<ActorMsgBus>::Get()->SendMsg(
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
    AsyncSendRegstMsgToProducer(in_regst);
    CHECK_EQ(0, inplace_consumed_rs_.TryPopFrontRegst(in_regst_desc_id));
  } else if (naive_produced_rs_.TryPushBackRegst(regst) != 0) {
    UpdtStateAsCustomizedProducedRegst(regst);
  }

  int64_t& expected_act_id = produced_regst2expected_act_id_[regst->regst_desc_id()];
  if (expected_act_id >= 0 && CheckOutputActId(regst->regst_desc_id())) {
    CHECK_EQ(regst->act_id(), expected_act_id);
  }
  expected_act_id = regst->act_id() + ActNumForEachOutput(regst->regst_desc_id());
  return 0;
}

void Actor::EnqueueAsyncMsg(const ActorMsg& msg) {
  if (is_kernel_launch_synchronized_
      && GetGlobalWorkStreamId()
             == Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg.dst_actor_id())) {
    Global<ActorMsgBus>::Get()->SendMsg(msg);
  } else {
    async_msg_queue_.push_back(msg);
  }
}

int64_t Actor::GetGlobalWorkStreamId() const {
  return Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(actor_id_);
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

std::unique_ptr<Actor> NewActor(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  Actor* rptr = NewObj<int32_t, Actor>(task_proto.task_type());
  const auto& job_descs = *Global<RuntimeJobDescs>::Get();
  rptr->Init(&job_descs.job_desc(task_proto.job_id()), task_proto, thread_ctx);
  return std::unique_ptr<Actor>(rptr);
}

void Actor::AsyncSendQueuedMsg() {
  if (!async_msg_queue_.empty()) {
    std::deque<ActorMsg> msgs;
    msgs.swap(async_msg_queue_);
    device_ctx_->AddCallBack([msgs]() {
      for (const ActorMsg& msg : msgs) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
    });
  }
}

}  // namespace oneflow
