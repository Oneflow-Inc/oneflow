#include "oneflow/core/actor/actor.h"

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
         || (model_version_id + 1)
                    % Global<JobDesc>::Get()->NumOfBatchesInSnapshot()
                == 0;
}

void Actor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  actor_id_ = task_proto.task_id();
  act_id_ = -1;
  if (task_proto.has_parallel_ctx()) {
    parallel_ctx_.reset(new ParallelContext(task_proto.parallel_ctx()));
  }
  for (const ExecNodeProto& node : task_proto.exec_sequence().exec_node()) {
    ExecKernel ek;
    ek.kernel = ConstructKernel(parallel_ctx(), node.kernel_conf());
    ek.bn_in_op2regst_desc_id = PbMap2HashMap(node.bn_in_op2regst_desc_id());
    exec_kernel_vec_.push_back(std::move(ek));
  }
  for (const auto& pair : task_proto.produced_regst_desc()) {
    Global<RegstMgr>::Get()->NewRegsts(
        pair.second, GetDeviceType(), task_proto.record_type(),
        [this](Regst* regst) {
          produced_regsts_[regst->regst_desc_id()].emplace_back(regst);
        });
    int64_t regst_desc_id = pair.second.regst_desc_id();
    CHECK(name2regst_desc_id_.emplace(pair.first, regst_desc_id).second);
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK(name2regst_desc_id_.emplace(pair.first, pair.second).second);
  }
  msg_handler_ = nullptr;
  for (const auto& pair : produced_regsts_) {
    for (const auto& regst : pair.second) {
      writeable_produced_regst_[regst->regst_desc_id()].push_back(regst.get());
      produced_regst2reading_cnt_[regst.get()] = 0;
    }
  }
  writeable_produced_regst_desc_num_ = writeable_produced_regst_.size();
  total_reading_cnt_ = 0;
  remaining_eord_cnt_ = task_proto.consumed_regst_desc_id().size();
  InitDeviceCtx(thread_ctx);
  VirtualActorInit(task_proto);
}

int64_t Actor::machine_id() const {
  return Global<IDMgr>::Get()->MachineId4ActorId(actor_id_);
}

int64_t Actor::thrd_id() const {
  return Global<IDMgr>::Get()->ThrdId4ActorId(actor_id_);
}

int64_t Actor::RegstDescId4Name(const std::string& name) const {
  auto find_it = name2regst_desc_id_.find(name);
  if (find_it != name2regst_desc_id_.end()) { return find_it->second; }
  return -1;
}

void Actor::InitDeviceCtx(const ThreadCtx&) {
  switch (GetDeviceType()) {
    case DeviceType::kCPU: {
      device_ctx_.reset(new CpuDeviceCtx(GetReservedWorkStreamId(0)));
      break;
    }
#ifdef WITH_CUDA
    case DeviceType::kGPU: {
      device_ctx_.reset(new CudaDeviceCtx(
          NewWorkStreamId(), cuda_handle_.cuda_stream(),
          cuda_handle_.cublas_pmh_handle(), cuda_handle_.cublas_pmd_handle(),
          cuda_handle_.cudnn_handle(), cuda_handle_.eigen_gpu_device()));
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

void Actor::SetReadableRegstInfo(const Regst* regst, ReadableRegstInfo* info) {
  info->set_regst_desc_id(regst->regst_desc_id());
  info->set_act_id(regst->act_id());
}

int Actor::HandlerZombie(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    remaining_eord_cnt_ -= 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      AsyncSendRegstMsgToProducer(msg.regst());
    }
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
    act_id_ += 1;
    ActEvent* act_event = nullptr;
    if (Global<RuntimeCtx>::Get()->is_experiment_phase()) {
      act_event = new ActEvent;
      act_event->set_actor_id(actor_id_);
      act_event->set_act_id(act_id_);
      act_event->set_work_stream_id(device_ctx_->work_stream_id());
      ForEachCurReadableRegst([&](const Regst* readable_regst) {
        ReadableRegstInfo* info = act_event->add_readable_regst_infos();
        SetReadableRegstInfo(readable_regst, info);
      });
      device_ctx_->AddCallBack(
          [act_event]() { act_event->set_start_time(GetCurTime()); });
    }
    Act();
    if (Global<RuntimeCtx>::Get()->is_experiment_phase()) {
      device_ctx_->AddCallBack([act_event]() {
        act_event->set_stop_time(GetCurTime());
        Global<CtrlClient>::Get()->PushActEvent(*act_event);
        delete act_event;
      });
    }
  }
}

bool Actor::IsWriteReady() {
  return writeable_produced_regst_desc_num_ == writeable_produced_regst_.size();
}

void Actor::DecreaseRemainingEordCnt() { remaining_eord_cnt_ -= 1; }

int Actor::TrySwitchToZombieOrFinish() {
  if (IsReadAlwaysUnReadyFromNow()) {
    AsyncReturnAllReadableRegst();
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

void Actor::AsyncLaunchKernel(
    const KernelCtx& kernel_ctx,
    std::function<Regst*(int64_t)> Regst4RegstDescId) {
  for (const ExecKernel& ek : exec_kernel_vec_) {
    ek.kernel->Launch(kernel_ctx, [&](const std::string& bn_in_op) -> Blob* {
      auto regst_desc_id_it = ek.bn_in_op2regst_desc_id.find(bn_in_op);
      if (regst_desc_id_it == ek.bn_in_op2regst_desc_id.end()) {
        return nullptr;
      }
      Regst* regst = Regst4RegstDescId(regst_desc_id_it->second);
      const std::string& lbn = ek.kernel->Lbn4BnInOp(bn_in_op);
      return regst->GetBlobByLbn(lbn);
    });
  }
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<bool(Regst*)> RegstPreProcess,
    std::function<bool(int64_t)> IsAllowedActor) {
  int64_t this_actor_id = actor_id_;
  for (auto& pair : writeable_produced_regst_) {
    if (pair.second.empty()) { continue; }
    Regst* regst = pair.second.front();
    if (RegstPreProcess(regst) == false) { continue; }
    auto regst_reading_cnt_it = produced_regst2reading_cnt_.find(regst);
    CHECK_EQ(regst_reading_cnt_it->second, 0);
    for (int64_t consumer : regst->consumers_actor_id()) {
      if (!IsAllowedActor(consumer)) { continue; }
      total_reading_cnt_ += 1;
      regst_reading_cnt_it->second += 1;
      regst->set_act_id(act_id_);
      device_ctx_->AddCallBack([consumer, regst, this_actor_id]() {
        ActorMsg msg =
            ActorMsg::BuildRegstMsgToConsumer(this_actor_id, consumer, regst);
        Global<ActorMsgBus>::Get()->SendMsg(std::move(msg));
      });
    }
    if (!regst->consumers_actor_id().empty()) { pair.second.pop_front(); }
    if (pair.second.empty()) { writeable_produced_regst_desc_num_ -= 1; }
  }
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<bool(Regst*)> RegstPreProcess) {
  AsyncSendRegstMsgToConsumer(RegstPreProcess, [](int64_t) { return true; });
}

void Actor::AsyncSendRegstMsgToConsumer(
    std::function<bool(int64_t)> IsAllowedActor) {
  AsyncSendRegstMsgToConsumer([](Regst*) { return true; }, IsAllowedActor);
}

void Actor::AsyncSendRegstMsgToConsumer() {
  AsyncSendRegstMsgToConsumer([](Regst*) { return true; });
}

void Actor::AsyncSendEORDMsgToConsumers(int64_t regst_desc_id) {
  const RtRegstDesc* regst_desc =
      produced_regsts_.at(regst_desc_id).front()->regst_desc();
  device_ctx_->AddCallBack([regst_desc]() {
    for (int64_t consumer : regst_desc->consumers_actor_id()) {
      ActorMsg msg =
          ActorMsg::BuildEordMsg(consumer, regst_desc->regst_desc_id());
      Global<ActorMsgBus>::Get()->SendMsg(std::move(msg));
    }
  });
}

void Actor::AsyncSendEORDMsgForAllProducedRegstDesc() {
  for (const auto& pair : produced_regsts_) {
    AsyncSendEORDMsgToConsumers(pair.first);
  }
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst) {
  AsyncSendRegstMsgToProducer(regst, regst->producer_actor_id());
}

void Actor::AsyncSendRegstMsgToProducer(Regst* regst, int64_t producer) {
  ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(actor_id_, producer, regst);
  device_ctx_->AddCallBack(
      [msg]() { Global<ActorMsgBus>::Get()->SendMsg(msg); });
}

void Actor::AsyncDo(std::function<void()> func) {
  device_ctx_->AddCallBack(func);
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
  writeable_it->second.push_back(regst);
  return 0;
}

Regst* Actor::GetCurWriteableRegst(int64_t regst_desc_id) {
  auto it = writeable_produced_regst_.find(regst_desc_id);
  if (it == writeable_produced_regst_.end()) { return nullptr; }
  if (it->second.empty()) { return nullptr; }
  return it->second.front();
}

Regst* Actor::GetCurWriteableRegst(const std::string& name) {
  return GetCurWriteableRegst(RegstDescId4Name(name));
}

Regst* Actor::GetCurSoleWriteableRegst() {
  CHECK_EQ(writeable_produced_regst_.size(), 1);
  return writeable_produced_regst_.begin()->second.front();
}

int64_t Actor::GetReservedWorkStreamId(int64_t reserved_id) {
  return Global<IDMgr>::Get()->GetReservedWorkStreamId(machine_id(), thrd_id(),
                                                       reserved_id);
}

int64_t Actor::NewWorkStreamId() {
  return Global<IDMgr>::Get()->NewWorkStreamId(machine_id(), thrd_id());
}

DeviceType Actor::GetDeviceType() const {
  return Global<IDMgr>::Get()->GetDeviceTypeFromActorId(actor_id_);
}

static HashMap<int, std::function<Actor*()>>& ActorCreatorMap() {
  static HashMap<int, std::function<Actor*()>> obj;
  return obj;
}

void AddActorCreator(TaskType task_type, std::function<Actor*()> creator) {
  CHECK(ActorCreatorMap().emplace(task_type, creator).second);
}

std::unique_ptr<Actor> NewActor(const TaskProto& task_proto,
                                const ThreadCtx& thread_ctx) {
  auto it = ActorCreatorMap().find(task_proto.task_type());
  CHECK(it != ActorCreatorMap().end()) << TaskType_Name(task_proto.task_type());
  Actor* rptr = it->second();
  rptr->Init(task_proto, thread_ctx);
  return std::unique_ptr<Actor>(rptr);
}

}  // namespace oneflow
