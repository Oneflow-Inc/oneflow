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
  is_naive_consumed_eord_ = false;
  TakeOverNaiveConsumed(task_proto.consumed_regst_desc_id());
  TakeOverNaiveProduced(task_proto.produced_regst_desc());
  VirtualActorInit(task_proto);
}

void Actor::TakeOverNaiveConsumed(const PbMap<std::string, RegstDescIdSet>& consumed_ids) {
  auto res = GetNaiveOrCustomizedConsumedRegstDescName();
  bool is_naive_names = res.first == RegstNameType::kNaive;
  const HashSet<std::string>& names = res.second;

  for (const auto& pair : consumed_ids) {
    bool find_the_name = names.find(pair.first) != names.end();
    if (is_naive_names == find_the_name || pair.first == "in_ctrl") {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
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
  int64_t pid = -1;
  naive_consumed_rs_.ForChosenFrontRegst(
      [&pid](int64_t) { return pid == -1; },
      [&pid](Regst* regst) {
        if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          pid = regst->piece_id();
        }
      });
  CHECK_NE(-1, pid);
  return pid;
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
      device_ctx_.reset(new CudaDeviceCtx(cuda_handle));
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

void Actor::ForEachCurNaiveReadableDataRegst(std::function<void(const Regst*)> func) const {
  naive_consumed_rs_.ForEachFrontRegst([func](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) { func(regst); }
  });
}

int Actor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
    CHECK(eord_regst_desc_ids_.insert(msg.eord_regst_desc_id()).second);
    if (naive_consumed_rs_.HasRegstDescId(msg.eord_regst_desc_id())) {
      is_naive_consumed_eord_ = true;
    } else {
      NormalProcessCustomizedEordMsg(msg);
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == Global<MachineCtx>::Get()->this_machine_id()) {
      Regst* regst = msg.regst();
      if (naive_consumed_rs_.HasRegstDescId(regst->regst_desc_id())) {
        CHECK_EQ(0, naive_consumed_rs_.TryPushBackRegst(regst));
        const auto& rdeq = naive_consumed_rs_.RegstDeq4RegstDescId(regst->regst_desc_id());
        CHECK(rdeq.empty() == false);
        if (rdeq.front()->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          NormalProcessNaiveReadableDataRegstMsg(rdeq);
        }
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
  if ((naive_consumed_rs_.total_regst_desc_cnt() != 0 && is_naive_consumed_eord_
       && naive_consumed_rs_.available_regst_desc_cnt() == 0)
      || (naive_consumed_rs_.total_regst_desc_cnt() == 0
          && IsCustomizedReadAlwaysUnReadyFromNow())) {
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
      Global<ThreadMgr>::Get()->compute_thread_pool()->AddWork(
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

    AsyncSendCustomizedConsumedRegstMsgToProducer();
    AsyncSendNaiveConsumedRegstMsgToProducer();
  }
}

void Actor::AsyncSendNaiveProducedRegstMsgToConsumer() {
  VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  AsyncSendProducedCtrlRegstMsgToConsumer();
}

void Actor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer();
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

  std::vector<int64_t> regst_desc_ids;
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
      regst_desc_ids.push_back(regst->regst_desc_id());
      AsyncSendMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, regst->producer_actor_id(), regst));
    }
  });
  naive_consumed_rs_.PopFrontRegsts(regst_desc_ids);
}

void Actor::AsyncSendProducedCtrlRegstMsgToConsumer() {
  auto IsChosenRegstDescId = [this](int64_t regst_desc_id) {
    return IsProducedCtrlRegstDescId(regst_desc_id) && ProducedCtrlRegstValid(regst_desc_id);
  };

  std::vector<int64_t> regst_desc_ids;
  naive_produced_rs_.ForChosenFrontRegst(IsChosenRegstDescId, [&](Regst* regst) {
    CHECK(regst->regst_desc()->regst_desc_type().has_ctrl_regst_desc());
    int64_t real_consumer_cnt = HandleRegstToConsumer(regst, [](int64_t) { return true; });
    if (real_consumer_cnt > 0) { regst_desc_ids.push_back(regst->regst_desc_id()); }
  });
  naive_produced_rs_.PopFrontRegsts(regst_desc_ids);
}

int64_t Actor::HandleRegstToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor) {
  auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
  CHECK_EQ(regst_reading_cnt_it->second, 0);
  regst->set_act_id(act_id_);

  int64_t real_consumer_cnt = 0;
  for (int64_t consumer : regst->consumers_actor_id()) {
    if (!IsAllowedActor(consumer)) { continue; }
    AsyncSendMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id_, consumer, regst));
    real_consumer_cnt += 1;
  }
  total_reading_cnt_ += real_consumer_cnt;
  regst_reading_cnt_it->second += real_consumer_cnt;
  return real_consumer_cnt;
}

bool Actor::IsReadReady() { return naive_consumed_rs_.IsCurSlotReady() && IsCustomizedReadReady(); }

bool Actor::IsWriteReady() {
  return naive_produced_rs_.IsCurSlotReady() && IsCustomizedWriteReady();
}

void Actor::AsyncLaunchKernel(const KernelCtx& kernel_ctx,
                              std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) { return nullptr; }
      Regst* regst = GetNaiveCurWriteable(regst_desc_id_it->second);
      if (regst == nullptr) { regst = GetNaiveCurReadable(regst_desc_id_it->second); }
      if (regst == nullptr) { regst = Regst4RegstDescId(regst_desc_id_it->second); }
      if (regst == nullptr) { return nullptr; }
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

void Actor::HandleProducedNaiveDataRegstToConsumer(std::function<bool(Regst*)> RegstPreProcess,
                                                   std::function<bool(int64_t)> IsAllowedActor) {
  std::vector<int64_t> regst_desc_ids;
  naive_produced_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (RegstPreProcess(regst) == false) { return; }
      int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
      if (real_consumer_cnt > 0) { regst_desc_ids.push_back(regst->regst_desc_id()); }
    }
  });
  naive_produced_rs_.PopFrontRegsts(regst_desc_ids);
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

void Actor::AsyncSendRegstMsgToConsumer(Regst* regst) {
  AsyncSendRegstMsgToConsumer(regst, [](int64_t) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(Regst* regst, std::function<bool(int64_t)> IsAllowedActor) {
  int64_t real_consumer_cnt = HandleRegstToConsumer(regst, IsAllowedActor);
  if (real_consumer_cnt > 0) { naive_produced_rs_.TryPopFrontRegst(regst->regst_desc_id()); }
}

void Actor::HandleConsumedNaiveDataRegstToProducer(std::function<bool(Regst*)> IsAllowedRegst) {
  naive_consumed_rs_.ForEachFrontRegst([&](Regst* regst) {
    if (regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
      if (IsAllowedRegst(regst) == false) { return; }
      AsyncSendRegstMsgToProducer(regst);
    }
  });
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
  AsyncSendMsg(ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst));
  naive_consumed_rs_.TryPopFrontRegst(regst_desc_id);
}

Regst* Actor::GetSoleProducedRegst4RegstDescId(int64_t regst_desc_id) {
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

  if (naive_produced_rs_.TryPushBackRegst(regst) != 0) {
    UpdtStateAsCustomizedProducedRegst(regst);
  }

  int64_t& expected_act_id = produced_regst2expected_act_id_[regst->regst_desc_id()];
  if (expected_act_id >= 0 && CheckOutputActId(regst->regst_desc_id())) {
    CHECK_EQ(regst->act_id(), expected_act_id);
  }
  expected_act_id = regst->act_id() + ActNumForEachOutput(regst->regst_desc_id());
  return 0;
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
