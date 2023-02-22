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

#include "oneflow/core/lazy/actor/actor_base.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/lazy/actor/actor_message.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/runtime_job_descs.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/core/lazy/stream_context/include/stream_context.h"

#ifdef WITH_CUDA

#include "oneflow/core/ep/cuda/cuda_stream.h"

#endif  // WITH_CUDA

namespace oneflow {

namespace {

enum RegstType : int8_t {
  kInvalid = 0,
  kProduced,
  kConsumed,
};

template<typename IndexType>
struct ProducedRegstState {
  IndexType reading_cnt;
  IndexType max_reading_cnt;
};

struct ConsumedRegstState {
  bool ready;
  bool eord;
};

template<typename IndexType>
struct RegstState {
  Regst* regst;
  RegstType regst_type;
  union {
    ProducedRegstState<IndexType> produced;
    ConsumedRegstState consumed;
  };
};

struct KernelInfo {
  std::unique_ptr<const Kernel> kernel;
  HashMap<std::string, Blob*> bn_in_op2blob;
  std::shared_ptr<KernelState> state;
};

template<typename IndexType, int max_size>
struct ArrayBaseIndex {
  ArrayBaseIndex() { std::memset(this, 0, sizeof(*this)); }

  inline IndexType Size() const { return size; }

  void Reserve(IndexType new_size) { CHECK_LE(new_size, max_size); }

  inline IndexType Lookup(int64_t v) const {
    for (IndexType i = 0; i < size; ++i) {
      if (arr[i] == v) { return i; }
    }
    CHECK(false);
    return -1;
  }

  bool Contains(int64_t v) const {
    for (IndexType i = 0; i < size; ++i) {
      if (arr[i] == v) { return true; }
    }
    return false;
  }

  IndexType Add(int64_t v) {
    CHECK_LT(size, max_size);
    const IndexType index = size;
    size += 1;
    arr[index] = v;
    return index;
  }

  void GetValues(std::vector<int64_t>* values) const {
    values->resize(size);
    for (IndexType i = 0; i < size; ++i) { values->at(i) = arr[i]; }
  }

  std::array<int64_t, max_size> arr;
  IndexType size;
};

template<typename IndexType>
struct MapBaseIndex {
  inline IndexType Size() const { return index_map.size(); }

  void Reserve(IndexType size) { index_map.reserve(size); }

  inline IndexType Lookup(int64_t v) {
    auto it = index_map.find(v);
    CHECK(it != index_map.end());
    return it->second;
  }

  bool Contains(int64_t v) { return index_map.count(v) > 0; }

  IndexType Add(int64_t v) {
    const IndexType index = index_map.size();
    CHECK(index_map.emplace(v, index).second);
    return index;
  }

  void GetValues(std::vector<int64_t>* values) const {
    values->resize(index_map.size());
    for (const auto& pair : index_map) { values->at(pair.second) = pair.first; }
  }

  HashMap<int64_t, IndexType> index_map;
};

template<typename IndexType, int max_size>
struct ArrayBaseStateContainer {
  ArrayBaseStateContainer() { std::memset(this, 0, sizeof(*this)); }

  void Resize(IndexType new_size) {
    CHECK_LE(new_size, max_size);
    size = new_size;
  }

  inline IndexType Size() const { return size; }

  inline RegstState<IndexType>& Get(IndexType index) {
    CHECK_LT(index, size);
    return arr[index];
  }

  std::array<RegstState<IndexType>, max_size> arr;
  IndexType size;
};

template<typename IndexType>
struct VectorBaseStateContainer {
  void Resize(IndexType new_size) { vec.resize(new_size); }

  inline IndexType Size() const { return static_cast<IndexType>(vec.size()); }

  inline RegstState<IndexType>& Get(IndexType index) { return vec.at(index); }

  std::vector<RegstState<IndexType>> vec;
};

bool IsInplaceRegstDesc(const RegstDescProto& regst_desc) {
  return regst_desc.has_inplace_consumed_regst_desc_id() && regst_desc.consumer_task_id_size() > 0;
}

size_t GetRegstDescCount(const TaskProto& task) {
  size_t regst_cnt = task.produced_regst_desc().size();
  for (const auto& pair : task.consumed_regst_desc_id()) {
    regst_cnt += pair.second.regst_desc_id_size();
  }
  return regst_cnt;
}

size_t GetConsumerCount(const TaskProto& task) {
  size_t consumer_cnt = 0;
  for (const auto& pair : task.produced_regst_desc()) {
    consumer_cnt += pair.second.consumer_task_id_size();
  }
  return consumer_cnt;
}

bool NeedExecKernelWhenInplace(const TaskProto& task) {
  int64_t data_regst_cnt = 0;
  for (const auto& pair : task.produced_regst_desc()) {
    if (pair.second.regst_desc_type().has_data_regst_desc()) {
      if (data_regst_cnt != 0) { return true; }
      data_regst_cnt += 1;
      const DataRegstDesc& regst_desc = pair.second.regst_desc_type().data_regst_desc();
      if (regst_desc.lbi2blob_desc().size() != 1) { return true; }
      if (regst_desc.lbi2blob_desc().begin()->blob_desc().is_dynamic()) { return true; }
    }
  }
  if (data_regst_cnt != 1) { return true; }
  if (task.exec_sequence().exec_node().size() != 1) { return true; }
  const OperatorConf& op_conf =
      task.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  if (!op_conf.has_user_conf()) { return true; }
  const std::string& op_type = op_conf.user_conf().op_type_name();
  const bool is_const_inplace_op_type = (op_type == "expand_dims") || (op_type == "squeeze")
                                        || (op_type == "reshape") || (op_type == "reshape_like")
                                        || (op_type == "transpose");
  if (!is_const_inplace_op_type) { return true; }
  return false;
}

#ifdef WITH_CUDA_GRAPHS

bool IsCUDAGraphSupported(const Kernel* kernel) {
  auto* user_kernel = dynamic_cast<const UserKernel*>(kernel);
  return (user_kernel != nullptr && user_kernel->IsCudaGraphSupported());
}

#endif  // WITH_CUDA_GRAPHS

template<int exec_kernel, int inplace, typename IndexType, typename RegstIndex,
         typename StateContainer, bool dynamic_allocation>
class LightActor : public ActorBase, public KernelContext, public ActorContextProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LightActor);
  explicit LightActor(ActorContext* actor_ctx)
      : thread_(nullptr),
        actor_ctx_(actor_ctx),
        stream_ctx_(actor_ctx->stream_ctx()),
        stream_kernel_observer_(nullptr) {
    auto* kernel_observer_provider = dynamic_cast<KernelObserverProvider*>(stream_ctx_);
    if (kernel_observer_provider != nullptr) {
      stream_kernel_observer_ = kernel_observer_provider->GetKernelObserver();
    }
  }
  ~LightActor() override {
    for (IndexType i = 0; i < index2state_.Size(); ++i) {
      auto& state = index2state_.Get(i);
      if (state.regst_type == RegstType::kProduced) { delete state.regst; }
    }
  }

  void Init(const JobDesc* job_desc, ActorContext* actor_ctx) override {
    const TaskProto& task_proto = actor_ctx->task_proto();
    CHECK_EQ(task_proto.exec_sequence().exec_node_size(), 1);
    if (exec_kernel) {
      kernel_info_[0].reset(new KernelInfo());
      const KernelConf& kernel_conf = task_proto.exec_sequence().exec_node(0).kernel_conf();
      kernel_info_[0]->kernel = ConstructKernel(kernel_conf, this);
#ifdef WITH_CUDA_GRAPHS
      auto* cuda_stream = dynamic_cast<ep::CudaStream*>(actor_ctx->stream_ctx()->stream());
      if (cuda_stream != nullptr && kernel_conf.all_blobs_are_static()
          && IsCUDAGraphSupported(kernel_info_[0]->kernel.get())) {
        cuda_graph_exec_[0].reset(new ep::CudaGraphExecutable());
      }
#endif
    }
    const int64_t thrd_id = ThrdId4ActorId(task_proto.task_id());
    thread_ = Singleton<ThreadMgr>::Get()->GetThrd(thrd_id);
    total_reading_cnt_ = 0;
    max_total_reading_cnt_ = 0;
    remaining_eord_cnt_ = 0;
    ready_consumed_ = 0;
    max_ready_consumed_ = 0;

    const IndexType regst_cnt = GetRegstDescCount(task_proto);
    regst_desc_id_index_.Reserve(regst_cnt);
    index2state_.Resize(regst_cnt);

    IndexType inplace_produced_index = -1;
    IndexType inplace_consumed_index = -1;
    int64_t inplace_consumed_regst_desc_id = -1;

    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst_desc = pair.second;
      if (IsInplaceRegstDesc(regst_desc)) {
        CHECK_EQ(inplace_consumed_regst_desc_id, -1);
        inplace_consumed_regst_desc_id = regst_desc.inplace_consumed_regst_desc_id();
      }
    }

    for (const auto& pair : task_proto.consumed_regst_desc_id()) {
      for (int64_t regst_desc_id : pair.second.regst_desc_id()) {
        const IndexType index = regst_desc_id_index_.Add(regst_desc_id);
        auto& state = index2state_.Get(index);
        state.regst_type = RegstType::kConsumed;
        state.consumed.ready = false;
        state.consumed.eord = false;
        remaining_eord_cnt_ += 1;
        max_ready_consumed_ += 1;
        if (regst_desc_id == inplace_consumed_regst_desc_id) { inplace_consumed_index = index; }
      }
    }

    for (const auto& pair : task_proto.produced_regst_desc()) {
      const RegstDescProto& regst_desc = pair.second;
      const IndexType index = regst_desc_id_index_.Add(regst_desc.regst_desc_id());
      auto& state = index2state_.Get(index);

      Singleton<RegstMgr>::Get()->NewRegsts(regst_desc, [&state](Regst* regst) {
        CHECK(state.regst == nullptr);
        state.regst = regst;
      });
      state.produced.max_reading_cnt = regst_desc.consumer_task_id_size();
      state.regst_type = RegstType::kProduced;
      state.produced.reading_cnt = 0;
      max_total_reading_cnt_ += state.produced.max_reading_cnt;
      if (IsInplaceRegstDesc(regst_desc)) {
        CHECK_EQ(inplace_produced_index, -1);
        inplace_produced_index = index;
      }
    }

    if (inplace) {
      CHECK_NE(inplace_produced_index, -1);
      CHECK_NE(inplace_consumed_index, -1);
      inplace_produced_index_[0] = inplace_produced_index;
      inplace_consumed_index_[0] = inplace_consumed_index;
    } else {
      CHECK_EQ(inplace_produced_index, -1);
      CHECK_EQ(inplace_consumed_index, -1);
    }
  }

  int ProcessMsg(const ActorMsg& msg) override {
    HandleActorMsg(msg);
    if (total_reading_cnt_ != 0) { return 0; }
    if (ready_consumed_ == max_ready_consumed_) {
      ActOnce();
      return 0;
    }
    if (OF_PREDICT_FALSE(ready_consumed_ == 0 && remaining_eord_cnt_ == 0)) {
      SendEORDMsg();
      return 1;
    }
    return 0;
  }

 private:
  void InitBnInOp2Blob() {
    if (exec_kernel) {
      const ExecNodeProto& node = actor_ctx_->task_proto().exec_sequence().exec_node(0);
      for (auto& pair : node.kernel_conf().op_attribute().arg_signature().bn_in_op2lbi()) {
        const std::string& bn = pair.first;
        auto regst_desc_id_it = node.bn_in_op2regst_desc_id().find(bn);
        if (regst_desc_id_it == node.bn_in_op2regst_desc_id().end()) {
          CHECK(kernel_info_[0]->bn_in_op2blob.emplace(bn, nullptr).second);
          continue;
        }
        if (!regst_desc_id_index_.Contains(regst_desc_id_it->second)) {
          CHECK(kernel_info_[0]->bn_in_op2blob.emplace(bn, nullptr).second);
          continue;
        }
        Regst* regst =
            index2state_.Get(regst_desc_id_index_.Lookup(regst_desc_id_it->second)).regst;
        if (regst == nullptr) {
          LOG(WARNING) << "null regst found, op:"
                       << node.kernel_conf().op_attribute().op_conf().name();
          CHECK(kernel_info_[0]->bn_in_op2blob.emplace(bn, nullptr).second);
          continue;
        }
        Blob* blob = regst->GetBlobByLbi(pair.second);
        if (!blob) {
          LOG(WARNING) << "null blob found, op: "
                       << node.kernel_conf().op_attribute().op_conf().name();
        }
        CHECK(kernel_info_[0]->bn_in_op2blob.emplace(bn, blob).second);
      }
    }
  }

  void InitActMsg() {
    const bool is_kernel_launch_synchronized =
        (!exec_kernel) || kernel_info_[0]->kernel->IsKernelLaunchSynchronized();
    const int64_t actor_id = actor_ctx_->task_proto().task_id();
    const int64_t thrd_id = ThrdId4ActorId(actor_id);
    auto IsSyncMsg = [&](const ActorMsg& msg) {
      return is_kernel_launch_synchronized && thrd_id == ThrdId4ActorId(msg.dst_actor_id());
    };
    auto EnqueueActorMsg = [&](const ActorMsg& msg) {
      if (IsSyncMsg(msg)) {
        sync_post_act_msgs_.emplace_back(msg);
      } else {
        async_post_act_msgs_.emplace_back(msg);
      }
    };
    std::vector<int64_t> index2regst_desc_id;
    regst_desc_id_index_.GetValues(&index2regst_desc_id);
    for (IndexType i = 0; i < index2state_.Size(); ++i) {
      const auto& state = index2state_.Get(i);
      if (state.regst_type == RegstType::kProduced) {
        for (int64_t consumer : state.regst->consumers_actor_id()) {
          EnqueueActorMsg(ActorMsg::BuildRegstMsgToConsumer(actor_id, consumer, state.regst));
        }
      } else if (state.regst_type == RegstType::kConsumed) {
        const int64_t regst_desc_id = index2regst_desc_id.at(i);
        int64_t producer = -1;
        if (Singleton<RegstMgr>::Get()->HasProducerTaskId4RegstDescId(regst_desc_id)) {
          producer = Singleton<RegstMgr>::Get()->ProducerTaskId4RegstDescId(regst_desc_id);
        } else {
          producer = state.regst->producer_actor_id();
        }
        ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(actor_id, producer, state.regst);
        if (inplace && i == inplace_consumed_index_[0]) {
          if (IsSyncMsg(msg)) {
            return_inplace_consumed_fn_[0] = [this, msg]() { thread_->EnqueueActorMsg(msg); };
          } else {
            return_inplace_consumed_fn_[0] = [this, msg]() {
              actor_ctx_->AddCallback([msg] { Singleton<ActorMsgBus>::Get()->SendMsg(msg); });
            };
          }
        } else {
          EnqueueActorMsg(msg);
        }
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  inline void ResetState() {
    total_reading_cnt_ = max_total_reading_cnt_;
    ready_consumed_ = 0;
    for (IndexType i = 0; i < index2state_.Size(); ++i) {
      auto& state = index2state_.Get(i);
      if (state.regst_type == RegstType::kProduced) {
        state.produced.reading_cnt = state.produced.max_reading_cnt;
        if (dynamic_allocation && state.produced.max_reading_cnt == 0
            && state.regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
          if (state.regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
            if (inplace && i == inplace_produced_index_[0]) {
              // do nothing
            } else {
              CHECK_JUST(
                  actor_ctx_->stream_ctx()->stream()->FreeAsync(state.regst->body_mem_ptr()));
            }
            state.regst->ResetBodyMemPtr(nullptr);
          } else if (state.regst->allocation_type() == RegstAllocationType::kStatic) {
            // do nothing
          } else {
            UNIMPLEMENTED();
          }
        }
      } else if (state.regst_type == RegstType::kConsumed) {
        state.consumed.ready = false;
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  inline void HandleActorMsg(const ActorMsg& msg) {
    if (OF_PREDICT_TRUE(msg.msg_type() == ActorMsgType::kRegstMsg)) {
      HandleRegstMsg(msg);
    } else if (msg.msg_type() == ActorMsgType::kEordMsg) {
      HandleEordMsg(msg);
    } else if (msg.msg_type() == ActorMsgType::kCmdMsg) {
      CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
    } else {
      UNIMPLEMENTED() << msg.msg_type() << " " << actor_ctx_->task_proto().task_id();
    }
  }

  void HandleEordMsg(const ActorMsg& msg) {
    const IndexType index = regst_desc_id_index_.Lookup(msg.eord_regst_desc_id());
    auto& state = index2state_.Get(index);
    CHECK_EQ(state.regst_type, RegstType::kConsumed);
    CHECK_EQ(state.consumed.eord, false);
    state.consumed.eord = true;
    CHECK_GT(remaining_eord_cnt_, 0);
    remaining_eord_cnt_ -= 1;
  }

  inline void HandleRegstMsg(const ActorMsg& msg) {
    int64_t regst_desc_id = msg.regst_desc_id();
    if (regst_desc_id == -1) { regst_desc_id = msg.regst()->regst_desc_id(); }
    const IndexType index = regst_desc_id_index_.Lookup(regst_desc_id);
    auto& state = index2state_.Get(index);
    if (state.regst_type == RegstType::kProduced) {
      CHECK_GT(state.produced.reading_cnt, 0);
      state.produced.reading_cnt -= 1;
      CHECK_GT(total_reading_cnt_, 0);
      total_reading_cnt_ -= 1;

      if (dynamic_allocation && state.produced.reading_cnt == 0
          && state.regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
        if (state.regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
          if (inplace && index == inplace_produced_index_[0]) {
            // do nothing
          } else {
            CHECK_JUST(actor_ctx_->stream_ctx()->stream()->FreeAsync(state.regst->body_mem_ptr()));
          }
          state.regst->ResetBodyMemPtr(nullptr);
        } else if (state.regst->allocation_type() == RegstAllocationType::kStatic) {
          // do nothing
        } else {
          UNIMPLEMENTED();
        }
      }

      if (inplace && index == inplace_produced_index_[0] && state.produced.reading_cnt == 0) {
        return_inplace_consumed_fn_[0]();
      }
    } else if (state.regst_type == RegstType::kConsumed) {
      CHECK_EQ(state.consumed.ready, false);
      CHECK_EQ(state.consumed.eord, false);
      if (state.regst == nullptr) {
        state.regst = msg.regst();
      } else {
        CHECK(state.regst == msg.regst());
      }
      ready_consumed_ += 1;
    } else {
      UNIMPLEMENTED();
    }
  }

  inline void ActOnce() {
    if (OF_PREDICT_FALSE(sync_post_act_msgs_.empty() && async_post_act_msgs_.empty())) {
      InitBnInOp2Blob();
      InitActMsg();
    }

    for (IndexType i = 0; i < index2state_.Size(); ++i) {
      auto& state = index2state_.Get(i);
      if (dynamic_allocation && state.regst_type == RegstType::kProduced
          && state.regst->regst_desc()->regst_desc_type().has_data_regst_desc()) {
        if (state.regst->allocation_type() == RegstAllocationType::kStreamOrdered) {
          CHECK(state.regst->body_mem_ptr() == nullptr);
          void* body_ptr = nullptr;
          if (inplace && i == inplace_produced_index_[0]) {
            body_ptr = index2state_.Get(inplace_consumed_index_[0]).regst->body_mem_ptr();
          } else {
            CHECK_JUST(actor_ctx_->stream_ctx()->stream()->AllocAsync(
                &body_ptr, state.regst->regst_desc()->BodyByteSize4OneRegst()));
          }
          state.regst->ResetBodyMemPtr(body_ptr);
        } else if (state.regst->allocation_type() == RegstAllocationType::kStatic) {
          // do nothing
        } else {
          UNIMPLEMENTED();
        }
      }
    }

    if (exec_kernel) { LaunchKernel(); }
    ResetState();
    thread_->EnqueueActorMsg(sync_post_act_msgs_.cbegin(), sync_post_act_msgs_.cend());
    if (!async_post_act_msgs_.empty()) {
      actor_ctx_->AddCallback([this]() {
        for (const auto& msg : async_post_act_msgs_) {
          Singleton<ActorMsgBus>::Get()->SendMsg(msg);
        }
      });
    }
  }

  inline void LaunchKernel() {
#ifdef WITH_CUDA_GRAPHS
    bool is_capturing = false;
    if (cuda_graph_exec_[0]) {
      auto* cuda_stream = stream_ctx_->stream()->As<ep::CudaStream>();
      if (cuda_graph_exec_[0]->IsInstantiated()) {
        cuda_stream->LaunchGraph(cuda_graph_exec_[0].get());
        return;
      }
      auto* user_kernel =
          CHECK_NOTNULL(dynamic_cast<const UserKernel*>(kernel_info_[0]->kernel.get()));
      if (user_kernel->IsReadyForCudaGraphCapture(this)) {
        is_capturing = true;
        cuda_stream->BeginGraphCapture();
      }
    }
#endif
    kernel_info_[0]->kernel->Launch(this);
#ifdef WITH_CUDA_GRAPHS
    if (cuda_graph_exec_[0] && is_capturing) {
      auto* cuda_stream = stream_ctx_->stream()->As<ep::CudaStream>();
      cuda_stream->EndGraphCapture(cuda_graph_exec_[0].get());
      cuda_stream->LaunchGraph(cuda_graph_exec_[0].get());
    }
#endif
  }

  void SendEORDMsg() {
    for (IndexType i = 0; i < index2state_.Size(); ++i) {
      auto& state = index2state_.Get(i);
      if (state.regst_type != RegstType::kProduced) { continue; }
      const RtRegstDesc* regst_desc = state.regst->regst_desc();
      actor_ctx_->AddCallback([regst_desc]() {
        for (int64_t consumer : regst_desc->consumers_actor_id()) {
          Singleton<ActorMsgBus>::Get()->SendMsg(
              ActorMsg::BuildEordMsg(consumer, regst_desc->regst_desc_id()));
        }
      });
    }
  }

  ep::Stream* stream() const override { return stream_ctx_->stream(); }

  ActorContext* GetActorContext() const override { return actor_ctx_; }

  Blob* BnInOp2Blob(const std::string& bn) const override {
    if (exec_kernel) {
      auto it = kernel_info_[0]->bn_in_op2blob.find(bn);
      if (it == kernel_info_[0]->bn_in_op2blob.end()) {
        return nullptr;
      } else {
        return it->second;
      }
    } else {
      return nullptr;
    }
  }

  const std::shared_ptr<KernelState>& state() const override {
    if (exec_kernel) {
      return kernel_info_[0]->state;
    } else {
      static const std::shared_ptr<KernelState> null_state;
      return null_state;
    }
  }

  void set_state(std::shared_ptr<KernelState> state) override {
    CHECK(exec_kernel);
    kernel_info_[0]->state = std::move(state);
  }

  void WillForward(KernelContext* kernel_ctx, const Kernel* kernel) override {
    Singleton<KernelObserver>::Get()->WillForward(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->WillForward(kernel_ctx, kernel);
    }
  }

  void DidForward(KernelContext* kernel_ctx, const Kernel* kernel) override {
    CHECK_JUST_MSG(kernel_ctx->stream()->GetAsyncError(), kernel->op_conf().name());
    Singleton<KernelObserver>::Get()->DidForward(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->DidForward(kernel_ctx, kernel);
    }
  }

  void WillForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override {
    Singleton<KernelObserver>::Get()->WillForwardHeader(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->WillForwardHeader(kernel_ctx, kernel);
    }
  }

  void DidForwardHeader(KernelContext* kernel_ctx, const Kernel* kernel) override {
    Singleton<KernelObserver>::Get()->DidForwardHeader(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->DidForwardHeader(kernel_ctx, kernel);
    }
  }

  void WillForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override {
    Singleton<KernelObserver>::Get()->WillForwardDataContent(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->WillForwardDataContent(kernel_ctx, kernel);
    }
  }

  void DidForwardDataContent(KernelContext* kernel_ctx, const Kernel* kernel) override {
    Singleton<KernelObserver>::Get()->DidForwardDataContent(kernel_ctx, kernel);
    if (stream_kernel_observer_ != nullptr) {
      stream_kernel_observer_->DidForwardDataContent(kernel_ctx, kernel);
    }
  }

  RegstIndex regst_desc_id_index_;
  StateContainer index2state_;
  IndexType total_reading_cnt_;
  IndexType ready_consumed_;
  IndexType max_total_reading_cnt_;
  IndexType max_ready_consumed_;
  IndexType remaining_eord_cnt_;
  IndexType inplace_produced_index_[inplace];
  IndexType inplace_consumed_index_[inplace];
  std::function<void()> return_inplace_consumed_fn_[inplace];
  Thread* thread_;
  std::unique_ptr<KernelInfo> kernel_info_[exec_kernel];
#ifdef WITH_CUDA_GRAPHS
  std::unique_ptr<ep::CudaGraphExecutable> cuda_graph_exec_[exec_kernel];
#endif
  ActorContext* actor_ctx_;
  StreamContext* stream_ctx_;
  std::vector<ActorMsg> sync_post_act_msgs_;
  std::vector<ActorMsg> async_post_act_msgs_;
  KernelObserver* stream_kernel_observer_;
};

template<int kernel_exec, int inplace, typename IndexType, typename RegstIndex,
         typename StateContainer>
ActorBase* NewLightActor(ActorContext* actor_ctx) {
  const bool dynamic_allocation =
      ParseBooleanFromEnv("ONEFLOW_GRAPH_ENABLE_STREAM_ORDERED_MEMORY_ALLOCATION", false);
  if (dynamic_allocation) {
    return new LightActor<kernel_exec, inplace, IndexType, RegstIndex, StateContainer, true>(
        actor_ctx);
  } else {
    return new LightActor<kernel_exec, inplace, IndexType, RegstIndex, StateContainer, false>(
        actor_ctx);
  }
}

template<int kernel_exec, int inplace, typename IndexType>
ActorBase* DispatchNewLightActorMaxSize(ActorContext* actor_ctx) {
  const size_t regst_desc_count = GetRegstDescCount(actor_ctx->task_proto());
  if (regst_desc_count <= 2) {
    return NewLightActor<kernel_exec, inplace, IndexType, ArrayBaseIndex<IndexType, 2>,
                         ArrayBaseStateContainer<IndexType, 2>>(actor_ctx);
  } else if (regst_desc_count <= 4) {
    return NewLightActor<kernel_exec, inplace, IndexType, ArrayBaseIndex<IndexType, 4>,
                         ArrayBaseStateContainer<IndexType, 4>>(actor_ctx);
  } else if (regst_desc_count <= 8) {
    return NewLightActor<kernel_exec, inplace, IndexType, ArrayBaseIndex<IndexType, 8>,
                         ArrayBaseStateContainer<IndexType, 8>>(actor_ctx);
  } else {
    return NewLightActor<kernel_exec, inplace, IndexType, MapBaseIndex<IndexType>,
                         VectorBaseStateContainer<IndexType>>(actor_ctx);
  }
}

template<int kernel_exec, int inplace>
ActorBase* DispatchNewLightActorIndexType(ActorContext* actor_ctx) {
  size_t size = std::max(GetRegstDescCount(actor_ctx->task_proto()),
                         GetConsumerCount(actor_ctx->task_proto()));
  if (size <= static_cast<size_t>(std::numeric_limits<int8_t>::max())) {
    return DispatchNewLightActorMaxSize<kernel_exec, inplace, int8_t>(actor_ctx);
  } else if (size <= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return DispatchNewLightActorMaxSize<kernel_exec, inplace, int32_t>(actor_ctx);
  } else {
    return nullptr;
  }
}

template<int kernel_exec>
ActorBase* DispatchNewLightActorInplace(ActorContext* actor_ctx) {
  const auto& produced_regst_desc = actor_ctx->task_proto().produced_regst_desc();
  const size_t inplace_produced_regst_cnt =
      std::count_if(produced_regst_desc.cbegin(), produced_regst_desc.cend(),
                    [](const PbMapPair<std::string, RegstDescProto>& pair) {
                      return pair.second.has_inplace_consumed_regst_desc_id();
                    });
  if (inplace_produced_regst_cnt > 1) { return nullptr; }
  bool inplace = false;
  for (const auto& pair : produced_regst_desc) {
    const RegstDescProto& regst_desc = pair.second;
    if (IsInplaceRegstDesc(regst_desc)) {
      CHECK_EQ(inplace, false);
      inplace = true;
    }
  }
  if (inplace) {
    if (kernel_exec && NeedExecKernelWhenInplace(actor_ctx->task_proto())) {
      return DispatchNewLightActorIndexType<1, 1>(actor_ctx);
    } else {
      return DispatchNewLightActorIndexType<0, 1>(actor_ctx);
    }
  } else {
    return DispatchNewLightActorIndexType<kernel_exec, 0>(actor_ctx);
  }
}

ActorBase* NewLightActorWithKernel(ActorContext* actor_ctx) {
  return DispatchNewLightActorInplace<1>(actor_ctx);
}

ActorBase* NewLightActorWithoutKernel(ActorContext* actor_ctx) {
  return DispatchNewLightActorInplace<0>(actor_ctx);
}

ActorBase* TryNewLightActorWithoutInit(ActorContext* actor_ctx) {
  const TaskProto& task_proto = actor_ctx->task_proto();
  if (!task_proto.all_register_num_eq_one_hint()) { return nullptr; }
  if (task_proto.exec_sequence().exec_node_size() != 1) { return nullptr; }
  if (task_proto.task_type() == TaskType::kNormalForward) {
    const OperatorConf& op_conf =
        task_proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
    if (op_conf.has_variable_conf()) {
      return NewLightActorWithoutKernel(actor_ctx);
    } else {
      return NewLightActorWithKernel(actor_ctx);
    }
  } else if (task_proto.task_type() == TaskType::kCopyHd) {
    return NewLightActorWithKernel(actor_ctx);
  } else if (task_proto.task_type() == TaskType::kTick) {
    return NewLightActorWithoutKernel(actor_ctx);
  } else if (task_proto.task_type() == TaskType::kCollectiveBoxingGeneric) {
    return NewLightActorWithKernel(actor_ctx);
  } else {
    return nullptr;
  }
}

}  // namespace

std::unique_ptr<ActorBase> TryNewLightActor(ActorContext* actor_ctx) {
  ActorBase* actor = TryNewLightActorWithoutInit(actor_ctx);
  if (actor != nullptr) {
    const auto& job_descs = *Singleton<RuntimeJobDescs>::Get();
    actor->Init(&job_descs.job_desc(actor_ctx->task_proto().job_id()), actor_ctx);
  }
  return std::unique_ptr<ActorBase>(actor);
}

}  // namespace oneflow
