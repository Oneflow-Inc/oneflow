#include <oneflow/core/job/nccl_comm_manager.h>
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

bool IsFirstRegstInPieceWithOrder(const Regst* regst, ColIdOrder order) {
  return (order == ColIdOrder::kAscending && regst->col_id() == 0)
         || (order == ColIdOrder::kDescending && regst->IsMaxCol());
}

bool IsLastRegstInPieceWithOrder(const Regst* regst, ColIdOrder order) {
  return (order == ColIdOrder::kAscending && regst->IsMaxCol())
         || (order == ColIdOrder::kDescending && regst->col_id() == 0);
}

bool NeedModelSave(int64_t model_version_id) {
  return model_version_id + 1 == Global<JobDesc>::Get()->TotalBatchNum()
         || (model_version_id + 1) % Global<JobDesc>::Get()->NumOfBatchesInSnapshot() == 0;
}

void Actor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  TaskProto non_ctrl_task_proto = task_proto;
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
  remaining_eord_cnt_ = 0;
  msg_handler_ = nullptr;
  eord_regst_desc_ids_.clear();
  // ctrl regst
  for (const auto& pair : task_proto.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_ctrl_regst_desc()) {
      non_ctrl_task_proto.mutable_produced_regst_desc()->erase(pair.first);
      Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
        produced_ctrl_regst_[regst->regst_desc_id()].emplace_back(regst);
      });
    }
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    if (pair.first == "in_ctrl") {
      non_ctrl_task_proto.mutable_consumed_regst_desc_id()->erase(pair.first);
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        CHECK(consumed_ctrl_regst_.insert({regst_desc_id, {}}).second);
      }
      remaining_eord_cnt_ += pair.second.regst_desc_id_size();
    }
  }
  for (const auto& pair : produced_ctrl_regst_) {
    for (const auto& regst : pair.second) {
      writeable_produced_ctrl_regst_[regst->regst_desc_id()].push_back(regst.get());
      produced_ctrl_regst2reading_cnt_[regst.get()] = 0;
    }
    produced_ctrl_regst2expected_act_id_[pair.first] = act_id_;
  }

  // non ctrl regst
  for (const auto& pair : non_ctrl_task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(pair.second, [this](Regst* regst) {
      produced_data_regsts_[regst->regst_desc_id()].emplace_back(regst);
    });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.insert({pair.first, {regst_desc_id}}).second);
  }
  for (const auto& pair : non_ctrl_task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.find(pair.first) == name2regst_desc_id_.end());
    std::vector<int64_t>& regst_desc_id_vec = name2regst_desc_id_[pair.first];
    for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
      regst_desc_id_vec.push_back(regst_desc_id);
    }
    remaining_eord_cnt_ += pair.second.regst_desc_id_size();
  }
  for (const auto& pair : produced_data_regsts_) {
    for (const auto& regst : pair.second) {
      writeable_produced_data_regst_[regst->regst_desc_id()].push_back(regst.get());
      produced_data_regst2reading_cnt_[regst.get()] = 0;
    }
    produced_data_regst2expected_act_id_[pair.first] = act_id_;
  }
  actual_writeable_produced_data_regst_desc_num_ = writeable_produced_data_regst_.size();
  writeable_produced_data_regst_desc_cnt_ = actual_writeable_produced_data_regst_desc_num_;
  total_reading_data_cnt_ = 0;
  total_reading_ctrl_cnt_ = 0;
  naive_readable_data_regst_.clear();
  naive_readable_data_regst_cnt_ = 0;
  readable_ctrl_regst_desc_cnt_ = 0;
  writeable_ctrl_regst_desc_cnt_ = writeable_produced_ctrl_regst_.size();
  is_naive_readable_data_eord_ = false;
  is_consumed_ctrl_eord_ = false;
  TakeOverNaiveConsumed(non_ctrl_task_proto.consumed_regst_desc_id());
  VirtualActorInit(non_ctrl_task_proto);
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

const std::vector<int64_t>& Actor::Name2RegstDescId(const std::string& name) const {
  return name2regst_desc_id_.at(name);
}

void Actor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  switch (GetDeviceType()) {
    case DeviceType::kCPU: {
      CHECK_EQ(GetLocalWorkStreamId(), 0);
      device_ctx_.reset(new CpuDeviceCtx());
      break;
    }
    case DeviceType::kGPU: {
      CudaStreamHandle* cuda_handle = nullptr;
      CHECK_EQ(GetLocalWorkStreamId(), 0);
      cuda_handle = thread_ctx.g_cuda_stream.get();
      device_ctx_.reset(
          new CudaDeviceCtx(cuda_handle, Global<NcclCommMgr>::Get()->NcclComm4ActorId(actor_id_)));
      break;
    }
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

void Actor::ForEachCurNaiveReadableRegst(std::function<void(const Regst*)> func) const {
  for (const auto& pair : naive_readable_data_regst_) {
    if (pair.second.empty() == false) { func(pair.second.front()); }
  }
}

int Actor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
    CHECK(eord_regst_desc_ids_.insert(msg.eord_regst_desc_id()).second);
    if (naive_readable_data_regst_.find(msg.eord_regst_desc_id())
        != naive_readable_data_regst_.end()) {
      is_naive_readable_data_eord_ = true;
    } else if (consumed_ctrl_regst_.find(msg.eord_regst_desc_id()) != consumed_ctrl_regst_.end()) {
      is_consumed_ctrl_eord_ = true;
    } else {
      NormalProcessCustomizedEordMsg(msg);
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (ProcessWriteableCtrlRegstMsg(msg) == 0 || ProcessReadableCtrlRegstMsg(msg) == 0) {
      // do nothing
    } else if (msg.SrcMachineId() == Global<MachineCtx>::Get()->this_machine_id()) {
      Regst* regst = msg.regst();
      auto naive_readable_regst_it = naive_readable_data_regst_.find(regst->regst_desc_id());
      if (naive_readable_regst_it != naive_readable_data_regst_.end()) {
        if (naive_readable_regst_it->second.empty()) { naive_readable_data_regst_cnt_ += 1; }
        naive_readable_regst_it->second.push_back(regst);
        NormalProcessNaiveReadableRegstMsg(naive_readable_regst_it->second);
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
  // TODO: refactor code below for potential bugs
  if (((is_naive_readable_data_eord_ && naive_readable_data_regst_cnt_ == 0)
       || IsCustomizedReadAlwaysUnReadyFromNow())
      && ((is_consumed_ctrl_eord_ && readable_ctrl_regst_desc_cnt_ == 0)
          || consumed_ctrl_regst_.size() == 0)) {
    CHECK_EQ(naive_readable_data_regst_cnt_, 0);
    CHECK_EQ(readable_ctrl_regst_desc_cnt_, 0);
    AsyncReturnAllCustomizedReadableRegst();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    AsyncSendEORDMsgForAllProducedCtrlRegstDesc();
    if (remaining_eord_cnt_ == 0 && total_reading_data_cnt_ == 0 && total_reading_ctrl_cnt_ == 0) {
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
    if (ProcessWriteableCtrlRegstMsg(msg) != 0) {
      if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
        AsyncSendRegstMsgToProducer(msg.regst());
      }
    }
  } else {
    UNIMPLEMENTED();
  }
  if (remaining_eord_cnt_ == 0 && total_reading_data_cnt_ == 0 && total_reading_ctrl_cnt_ == 0) {
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
    ForEachCurNaiveReadableRegst([&](const Regst* readable_regst) {
      ReadableRegstInfo* info = act_event->add_readable_regst_infos();
      Actor::SetReadableRegstInfo(readable_regst, info);
    });
    ForEachCurCustomizedReadableRegst([&](const Regst* readable_regst) {
      ReadableRegstInfo* info = act_event->add_readable_regst_infos();
      SetReadableRegstInfo(readable_regst, info);
    });
    ForEachCurConsumedCtrlRegst([&](const Regst* consumed_ctrl_regst) {
      ReadableRegstInfo* info = act_event->add_readable_regst_infos();
      Actor::SetReadableRegstInfo(consumed_ctrl_regst, info);
    });
    device_ctx_->AddCallBack([act_event]() { act_event->set_start_time(GetCurTime()); });

    DoAct();

    device_ctx_->AddCallBack([act_event]() {
      act_event->set_stop_time(GetCurTime());
      // The stream poller thread is not allowed to perform blocking RPC call. Hence, the
      // RPC call is forwarded to the thread pool and will be executed there.
      Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork(
          [act_event]() { Global<CtrlClient>::Get()->PushActEvent(*act_event); });
    });
  } else {
    DoAct();
  }
}

void Actor::ActUntilFail() {
  while (IsReadReady() && IsWriteReady() && IsCtrlReady()) {
    act_id_ += 1;
    std::function<bool(Regst*)> IsNaiveAllowedReturnToProducer = [](Regst*) { return true; };
    TryLogActEvent([&] { Act(&IsNaiveAllowedReturnToProducer); });
    AsyncSendCtrlRegstMsg();
    for (auto& pair : naive_readable_data_regst_) {
      CHECK_EQ(pair.second.empty(), false);
      if (IsNaiveAllowedReturnToProducer(pair.second.front()) == false) { continue; }
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop_front();
      if (pair.second.empty()) { naive_readable_data_regst_cnt_ -= 1; }
    }
  }
}

bool Actor::IsWriteReady() {
  return writeable_produced_data_regst_desc_cnt_ == actual_writeable_produced_data_regst_desc_num_;
}

void Actor::AsyncLaunchKernel(const KernelCtx& kernel_ctx,
                              std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) { return nullptr; }
      Regst* regst = GetCurWriteableRegst(regst_desc_id_it->second);
      if (regst == nullptr) { regst = GetNaiveCurReadable(regst_desc_id_it->second); }
      if (regst == nullptr) { regst = Regst4RegstDescId(regst_desc_id_it->second); }
      const LogicalBlobId& lbi = ek.kernel->BnInOp2Lbi(bn_in_op);
      return regst->GetBlobByLbi(lbi);
    });
  }
}

void Actor::AsyncLaunchKernel(const KernelCtx& kernel_ctx) {
  AsyncLaunchKernel(kernel_ctx, [](int64_t) -> Regst* {
    UNIMPLEMENTED();
    return nullptr;
  });
}

void Actor::AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                        std::function<bool(int64_t)> IsAllowedActor) {
  for (auto& pair : writeable_produced_data_regst_) {
    if (pair.second.empty()) { continue; }
    Regst* regst = pair.second.front();
    if (RegstPreProcess(regst) == false) { continue; }
    auto regst_reading_cnt_it = produced_data_regst2reading_cnt_.find(regst);
    CHECK_EQ(regst_reading_cnt_it->second, 0);
    regst->set_act_id(act_id_);
    for (int64_t consumer : regst->consumers_actor_id()) {
      if (!IsAllowedActor(consumer)) { continue; }
      total_reading_data_cnt_ += 1;
      regst_reading_cnt_it->second += 1;
      AsyncSendMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
    }
    if (!regst->consumers_actor_id().empty()) { pair.second.pop_front(); }
    if (pair.second.empty()) { writeable_produced_data_regst_desc_cnt_ -= 1; }
  }
}

void Actor::AsyncSendRegstMsgToConsumer(std::function<bool(Regst*)> RegstPreProcess) {
  AsyncSendRegstMsgToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(std::function<bool(int64_t)> IsAllowedActor) {
  AsyncSendRegstMsgToConsumer([](Regst*) { return true; }, IsAllowedActor);
}

void Actor::AsyncSendRegstMsgToConsumer() {
  AsyncSendRegstMsgToConsumer([](Regst*) { return true; });
}

void Actor::AsyncSendEORDMsgToConsumers(int64_t regst_desc_id) {
  const RtRegstDesc* regst_desc = produced_data_regsts_.at(regst_desc_id).front()->regst_desc();
  device_ctx_->AddCallBack([regst_desc]() {
    for (int64_t consumer : regst_desc->consumers_actor_id()) {
      ActorMsg msg = ActorMsg::BuildEordMsg(consumer, regst_desc->regst_desc_id());
      Global<ActorMsgBus>::Get()->SendMsg(std::move(msg));
    }
  });
}

void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (const auto& pair : produced_data_regsts_) { AsyncSendEORDMsgToConsumers(pair.first); }
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst) {
  AsyncSendRegstMsgToProducer(regst, regst->producer_actor_id());
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst, int64_t producer) {
  AsyncSendMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst));
}

Regst* Actor::GetCurWriteableRegst(int64_t regst_desc_id) {
  auto it = writeable_produced_data_regst_.find(regst_desc_id);
  if (it == writeable_produced_data_regst_.end()) { return nullptr; }
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

Regst* Actor::GetCurWriteableRegst(const std::string& name) {
  return GetCurWriteableRegst(Name2SoleRegstDescId(name));
}

Regst* Actor::GetCurSoleWriteableRegst() {
  CHECK_EQ(writeable_produced_data_regst_.size(), 1);
  return writeable_produced_data_regst_.begin()->second.front();
}

std::pair<bool, std::vector<std::string>> Actor::GetNaiveConsumedRegstDescName() {
  return {false, {}};
}

Regst* Actor::GetNaiveCurReadable(int64_t regst_desc_id) {
  auto it = naive_readable_data_regst_.find(regst_desc_id);
  if (it != naive_readable_data_regst_.end() && it->second.empty() == false) {
    return it->second.front();
  } else {
    return nullptr;
  }
}

Regst* Actor::GetNaiveNextReadable(int64_t regst_desc_id) {
  auto it = naive_readable_data_regst_.find(regst_desc_id);
  if (it == naive_readable_data_regst_.end()) { return nullptr; }
  if (it->second.size() < 2) { return nullptr; }
  return it->second.at(1);
}

Regst* Actor::GetNaiveSoleCurReadable() {
  CHECK_EQ(naive_readable_data_regst_.size(), 1);
  return GetNaiveFirstCurReadable();
}

Regst* Actor::GetNaiveFirstCurReadable() {
  auto naive_readable_regst_it = naive_readable_data_regst_.begin();
  CHECK(naive_readable_regst_it != naive_readable_data_regst_.end());
  CHECK_EQ(naive_readable_regst_it->second.empty(), false);
  return naive_readable_regst_it->second.front();
}

Regst* Actor::GetSoleProducedRegst(int64_t regst_desc_id) {
  auto it = produced_data_regsts_.find(regst_desc_id);
  CHECK(it != produced_data_regsts_.end());
  CHECK_EQ(it->second.size(), 1);
  return it->second.front().get();
}

int64_t Actor::GetSoleProducedDataRegstDescId() const {
  CHECK_EQ(produced_data_regsts_.size(), 1);
  return produced_data_regsts_.begin()->first;
}

bool Actor::IsReadReady() {
  return naive_readable_data_regst_.size() == naive_readable_data_regst_cnt_
         && IsCustomizedReadReady();
}

bool Actor::IsCtrlReady() {
  return writeable_ctrl_regst_desc_cnt_ == writeable_produced_ctrl_regst_.size()
         && readable_ctrl_regst_desc_cnt_ == consumed_ctrl_regst_.size();
}

int Actor::ProcessWriteableCtrlRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  auto reading_cnt_it = produced_ctrl_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_ctrl_regst2reading_cnt_.end()) { return -1; }
  CHECK(produced_ctrl_regst_.find(regst->regst_desc_id()) != produced_ctrl_regst_.end());
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_ctrl_cnt_ -= 1;
  if (reading_cnt_it->second != 0) { return 0; }
  auto writeable_it = writeable_produced_ctrl_regst_.find(regst->regst_desc_id());
  CHECK(writeable_it != writeable_produced_ctrl_regst_.end());
  if (writeable_it->second.empty()) { writeable_ctrl_regst_desc_cnt_ += 1; }
  int64_t& expected_act_id = produced_ctrl_regst2expected_act_id_[regst->regst_desc_id()];
  if (expected_act_id >= 0 && CheckOutputActId(regst->regst_desc_id())) {
    CHECK_EQ(regst->act_id(), expected_act_id);
  }
  expected_act_id = regst->act_id() + ActNumForEachOutput(regst->regst_desc_id());
  writeable_it->second.push_back(regst);
  return 0;
}

int Actor::ProcessReadableCtrlRegstMsg(const ActorMsg& msg) {
  int64_t regst_desc_id = msg.regst_desc_id();
  auto consumed_it = consumed_ctrl_regst_.find(regst_desc_id);
  if (consumed_it != consumed_ctrl_regst_.end()) {
    if (consumed_it->second.empty()) { readable_ctrl_regst_desc_cnt_ += 1; }
    consumed_it->second.push_back(msg.regst());
    return 0;
  }
  return -1;
}

void Actor::AsyncSendCtrlRegstMsg() {
  for (auto& pair : consumed_ctrl_regst_) {
    CHECK(!pair.second.empty());
    int32_t returned_regst_num =
        pair.second.front()->regst_desc()->regst_desc_type().ctrl_regst_desc().returned_regst_num();
    CHECK_GE(returned_regst_num, 1);
    CHECK_GE(pair.second.size(), returned_regst_num);

    while (returned_regst_num--) {
      Regst* regst = pair.second.front();
      AsyncSendMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, regst->producer_actor_id(), regst));
      pair.second.pop_front();
    }
    if (pair.second.empty()) { --readable_ctrl_regst_desc_cnt_; }
  }
  for (auto& pair : writeable_produced_ctrl_regst_) {
    CHECK(!pair.second.empty());
    Regst* regst = pair.second.front();
    regst->set_act_id(act_id_);
    auto regst_reading_cnt_it = produced_ctrl_regst2reading_cnt_.find(regst);
    CHECK_EQ(regst_reading_cnt_it->second, 0);
    for (int64_t consumer : regst->consumers_actor_id()) {
      AsyncSendMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
      ++total_reading_ctrl_cnt_;
      regst_reading_cnt_it->second += 1;
    }
    pair.second.pop_front();
    if (pair.second.empty()) { --writeable_ctrl_regst_desc_cnt_; }
  }
}

void Actor::AsyncSendEORDMsgForAllProducedCtrlRegstDesc() {
  for (auto& pair : produced_ctrl_regst_) {
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

void Actor::ForEachCurConsumedCtrlRegst(std::function<void(const Regst*)> func) const {
  for (const auto& pair : consumed_ctrl_regst_) {
    if (pair.second.empty() == false) { func(pair.second.front()); }
  }
}

int Actor::TryUpdtStateAsProducedRegst(Regst* regst) {
  auto reading_cnt_it = produced_data_regst2reading_cnt_.find(regst);
  if (reading_cnt_it == produced_data_regst2reading_cnt_.end()) { return -1; }
  CHECK(produced_data_regsts_.find(regst->regst_desc_id()) != produced_data_regsts_.end());
  CHECK_GE(reading_cnt_it->second, 1);
  reading_cnt_it->second -= 1;
  total_reading_data_cnt_ -= 1;
  if (reading_cnt_it->second != 0) { return 0; }
  auto writeable_it = writeable_produced_data_regst_.find(regst->regst_desc_id());
  CHECK(writeable_it != writeable_produced_data_regst_.end());
  if (writeable_it->second.empty()) { writeable_produced_data_regst_desc_cnt_ += 1; }
  int64_t& expected_act_id = produced_data_regst2expected_act_id_[regst->regst_desc_id()];
  if (expected_act_id >= 0 && CheckOutputActId(regst->regst_desc_id())) {
    CHECK_EQ(regst->act_id(), expected_act_id);
  }
  expected_act_id = regst->act_id() + ActNumForEachOutput(regst->regst_desc_id());
  writeable_it->second.push_back(regst);
  return 0;
}

void Actor::TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  std::pair<bool, std::vector<std::string>> isall_or_names = GetNaiveConsumedRegstDescName();
  if (isall_or_names.first) {
    for (const auto& pair : consumed_ids) { AddNaiveConsumed(pair.second); }
  } else {
    for (const std::string& name : isall_or_names.second) {
      auto it = consumed_ids.find(name);
      if (it != consumed_ids.end()) { AddNaiveConsumed(it->second); }
    }
  }
}

void Actor::AddNaiveConsumed(const RegstDescIdSet& regst_desc_ids) {
  for (int64_t regst_desc_id : regst_desc_ids.regst_desc_id()) {
    naive_readable_data_regst_[regst_desc_id] = {};
  }
}

void Actor::AsyncSendMsg(const ActorMsg& msg) {
  std::function<void()> callback = [msg]() { Global<ActorMsgBus>::Get()->SendMsg(msg); };
  if (GetGlobalWorkStreamId()
      == Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(msg.dst_actor_id())) {
    callback();
  } else {
    device_ctx_->AddCallBack(callback);
  }
}

int64_t Actor::GetGlobalWorkStreamId() const {
  return Global<IDMgr>::Get()->GlobalWorkStreamId4ActorId(actor_id_);
}

int64_t Actor::GetLocalWorkStreamId() const {
  return Global<IDMgr>::Get()->LocalWorkStreamId4ActorId(actor_id_);
}

std::unique_ptr<Actor> NewActor(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  Actor* rptr = NewObj<Actor>(task_proto.task_type());
  rptr->Init(task_proto, thread_ctx);
  return std::unique_ptr<Actor>(rptr);
}

}  // namespace oneflow
