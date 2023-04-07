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
#include "oneflow/core/job/collective_boxing/executor_backend_manager.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/cambricon/collective_communication/cncl_util.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/common/mlu_guard.h"

#include <memory>
#include <utility>

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

static const int64_t kNumOfCommInCurProcess = 1;

cnclReduceOp_t GetCnclReduceOp(ReduceMethod reduce_method) {
  if (reduce_method == kReduceMethodSum) {
    return cnclReduceOp_t::cnclSum;
  } else {
    UNIMPLEMENTED();
  }
}

std::string GetCnclUniqueIdRpcKey(const std::string& name, int64_t stream_id) {
  return "CollectiveBoxingExecutorCnclUniqueIdRpcKey-" + name + "-" + std::to_string(stream_id);
}

struct CopyParams {
  void* dst;
  const void* src;
  int64_t count;
};

constexpr int64_t kMultiCopyParamsMaxSize = 128;
constexpr int64_t kMultiCopyAlignSize = 64;  // MLU need align 64

int64_t GetMultiCopyAlignedSize(int64_t size) {
  return ((size + kMultiCopyAlignSize - 1) / kMultiCopyAlignSize) * kMultiCopyAlignSize;
}

struct MultiCopyParams {
  MultiCopyParams() : count(0), params{} {}

  void Add(void* dst, const void* src, int64_t count) {
    CHECK_LT(this->count, kMultiCopyParamsMaxSize);
    params[this->count].dst = dst;
    params[this->count].src = src;
    params[this->count].count = count;
    this->count += 1;
  }

  int64_t count = 0;
  CopyParams params[kMultiCopyParamsMaxSize];
};

void MultiCopy(cnrtQueue_t stream, const MultiCopyParams& multi_params) {
  if (multi_params.count <= 0) { return; }
  CHECK_LE(multi_params.count, kMultiCopyParamsMaxSize);
  for (int64_t i = 0; i < multi_params.count; ++i) {
    OF_MLU_CHECK(cnrtMemcpyAsync(multi_params.params[i].dst,
                                 const_cast<void*>(multi_params.params[i].src),
                                 multi_params.params[i].count, stream, cnrtMemcpyDevToDev));
  }
  OF_MLU_CHECK(cnrtQueueSync(stream));
}

class CommRank final {
 public:
  OF_DISALLOW_COPY(CommRank);
  CommRank(int32_t device_id, int32_t global_rank, int32_t global_rank_count, int32_t local_rank,
           int32_t local_rank_count)
      : device_id_(device_id),
        global_rank_(global_rank),
        local_rank_(local_rank),
        cncl_comm_(nullptr) {}

  CommRank(CommRank&& rhs) noexcept {
    this->device_id_ = rhs.device_id_;
    this->global_rank_ = rhs.global_rank_;
    this->local_rank_ = rhs.local_rank_;
    this->cncl_comm_ = rhs.cncl_comm_;
    rhs.cncl_comm_ = nullptr;
  }

  ~CommRank() {
    if (cncl_comm_ != nullptr) {
      MluCurrentDeviceGuard guard(device_id_);
      OF_CNCL_CHECK(cnclFreeComm(cncl_comm_));
    }
  }

  int32_t device_id() const { return device_id_; }

  cnclComm_t cncl_comm() const { return cncl_comm_; }

  void InitRank(cnclCliqueId clique_id, int32_t global_rank_count) {
    MluCurrentDeviceGuard guard(device_id_);
    int dev_list[kNumOfCommInCurProcess] = {device_id_};
    int rank_list[kNumOfCommInCurProcess] = {global_rank_};
    OF_CNCL_CHECK(cnclInitComms(&cncl_comm_, kNumOfCommInCurProcess, dev_list, rank_list,
                                global_rank_count, &clique_id));
  }

 private:
  int32_t device_id_;
  int32_t global_rank_;
  int32_t local_rank_;
  cnclComm_t cncl_comm_;
};

class CommGroup final {
 public:
  OF_DISALLOW_COPY(CommGroup);
  CommGroup() = default;
  ~CommGroup() = default;
  CommGroup(CommGroup&& rhs) noexcept {
    rank_vec_.swap(rhs.rank_vec_);
    global_rank_count_ = rhs.global_rank_count_;
  }

  void InitGroup(const DeviceSet& device_set, const std::string& unique_name) {
    MluCurrentDeviceGuard guard;
    const int64_t this_machine_id = GlobalProcessCtx::Rank();
    global_rank_count_ = device_set.device_size();
    std::vector<int32_t> local_ranks;
    for (int32_t i = 0; i < global_rank_count_; ++i) {
      if (device_set.device(i).machine_id() == this_machine_id) { local_ranks.emplace_back(i); }
    }
    const int32_t local_rank_count = local_ranks.size();
    CHECK_GT(local_rank_count, 0);
    cnclCliqueId cncl_clique_id{};
    if (local_ranks.front() == 0) {
      OF_CNCL_CHECK(cnclGetCliqueId(&cncl_clique_id));
      if (local_rank_count != global_rank_count_) {
        Singleton<CtrlClient>::Get()->PushKV(unique_name, CnclCliqueIdToString(cncl_clique_id));
      }
    } else {
      Singleton<CtrlClient>::Get()->PullKV(unique_name, [&cncl_clique_id](const std::string& val) {
        CnclCliqueIdFromString(val, &cncl_clique_id);
      });
    }
    rank_vec_.reserve(local_rank_count);
    for (int32_t local_rank = 0; local_rank < local_ranks.size(); ++local_rank) {
      const int32_t global_rank = local_ranks.at(local_rank);
      const int32_t device_id = device_set.device(global_rank).device_id();
      MluCurrentDeviceGuard guard(device_id);
      rank_vec_.emplace_back(device_id, global_rank, global_rank_count_, local_rank,
                             local_rank_count);
      rank_vec_.at(local_rank).InitRank(cncl_clique_id, global_rank_count_);
    }
  }

  int32_t global_rank_count() const { return global_rank_count_; }

  int32_t local_rank_count() const { return rank_vec_.size(); }

  const CommRank& GetCommRank(int32_t local_rank) const { return rank_vec_.at(local_rank); }

 private:
  std::vector<CommRank> rank_vec_;
  int32_t global_rank_count_ = 0;
};

class StreamCtx {
 public:
  OF_DISALLOW_COPY(StreamCtx);
  StreamCtx(int32_t device_id, size_t fusion_buffer_size)
      : device_id_(device_id), fusion_buffer_size_(fusion_buffer_size) {
    MluCurrentDeviceGuard guard(device_id_);
    int least_priority;
    int greatest_priority;
    OF_MLU_CHECK(cnrtDeviceGetQueuePriorityRange(&least_priority, &greatest_priority));
    OF_MLU_CHECK(cnrtQueueCreateWithPriority(&stream_, 0, greatest_priority));
    void* data_ptr = (void*)fusion_buffer_;
    OF_MLU_CHECK(cnrtMalloc(&data_ptr, fusion_buffer_size_));
    cb_event_poller_ = std::thread(&StreamCtx::PollEvent, this);
  }
  ~StreamCtx() {
    cb_event_chan_.Close();
    cb_event_poller_.join();
    MluCurrentDeviceGuard guard(device_id_);
    OF_MLU_CHECK(cnrtQueueSync(stream_));
    OF_MLU_CHECK(cnrtQueueDestroy(stream_));
    OF_MLU_CHECK(cnrtFree(fusion_buffer_));
  }

  void PollEvent() {
    MluCurrentDeviceGuard guard(device_id_);
    while (true) {
      std::pair<cnrtNotifier_t, std::function<void()>> cb_event;
      ChannelStatus status = cb_event_chan_.Receive(&cb_event);
      if (status == kChannelStatusErrorClosed) { break; }
      CHECK_EQ(status, kChannelStatusSuccess);
      OF_MLU_CHECK(cnrtWaitNotifier(cb_event.first));
      cb_event.second();
      OF_MLU_CHECK(cnrtNotifierDestroy(cb_event.first));
    }
  }

  void AddCallback(const std::function<void()>& callback) {
    MluCurrentDeviceGuard guard(device_id_);
    cnrtNotifier_t event;
    OF_MLU_CHECK(cnrtNotifierCreateWithFlags(&event, CNRT_NOTIFIER_DEFAULT));
    OF_MLU_CHECK(cnrtPlaceNotifier(event, stream_));
    CHECK_EQ(cb_event_chan_.Send(std::make_pair(event, callback)), kChannelStatusSuccess);
  }

  int32_t device_id() const { return device_id_; }

  cnrtQueue_t stream() const { return stream_; }

  size_t fusion_buffer_size() const { return fusion_buffer_size_; }

  char* fusion_buffer() const { return fusion_buffer_; }

 private:
  int32_t device_id_;
  cnrtQueue_t stream_ = nullptr;
  size_t fusion_buffer_size_;
  char* fusion_buffer_ = nullptr;
  Channel<std::pair<cnrtNotifier_t, std::function<void()>>> cb_event_chan_;
  std::thread cb_event_poller_;
};

void LaunchFusedAllReduce(const CommGroup& comm_group,
                          const std::vector<std::unique_ptr<StreamCtx>>& device_id2stream_ctx,
                          const std::shared_ptr<RequestStore>& request_store,
                          const std::vector<RequestId>& request_ids) {
  CHECK_LE(request_ids.size(), kMultiCopyParamsMaxSize);
  RequestEntry* first_request_entry = request_store->MutRequestEntry(request_ids.front());
  const cnclDataType_t cncl_data_type =
      GetCnclDataType(first_request_entry->desc().op_desc().data_type());
  const cnclReduceOp_t cncl_reduce_op =
      GetCnclReduceOp(first_request_entry->desc().op_desc().reduce_method());
  const int64_t size_of_data_type =
      GetSizeOfDataType(first_request_entry->desc().op_desc().data_type());
  std::vector<int64_t> offset_vec;
  offset_vec.reserve(request_ids.size());
  int64_t offset = 0;
  request_store->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
        offset_vec.emplace_back(offset);
        offset += GetMultiCopyAlignedSize(request_entry->size_in_bytes());
      });
  const int64_t elem_cnt = offset / size_of_data_type;
  for (int32_t local_rank = 0; local_rank < comm_group.local_rank_count(); ++local_rank) {
    MultiCopyParams copy_in_params;
    const CommRank& comm_rank = comm_group.GetCommRank(local_rank);
    const StreamCtx* stream_ctx = device_id2stream_ctx.at(comm_rank.device_id()).get();
    CHECK_LE(offset, stream_ctx->fusion_buffer_size());
    request_store->ForEachMutRequestEntryForIdsInJob(
        request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          copy_in_params.Add(stream_ctx->fusion_buffer() + offset_vec.at(i),
                             request_entry->GetRuntimeRequest(local_rank)->send_buff,
                             request_entry->size_in_bytes());
        });
    MluCurrentDeviceGuard guard(comm_rank.device_id());
    MultiCopy(stream_ctx->stream(), copy_in_params);
  }

  for (int32_t local_rank = 0; local_rank < comm_group.local_rank_count(); ++local_rank) {
    const CommRank& comm_rank = comm_group.GetCommRank(local_rank);
    const StreamCtx* stream_ctx = device_id2stream_ctx.at(comm_rank.device_id()).get();
    MluCurrentDeviceGuard guard((comm_rank.device_id()));
    OF_CNCL_CHECK(cnclAllReduce(stream_ctx->fusion_buffer(), stream_ctx->fusion_buffer(), elem_cnt,
                                cncl_data_type, cncl_reduce_op, comm_rank.cncl_comm(),
                                stream_ctx->stream()));
  }

  for (int32_t local_rank = 0; local_rank < comm_group.local_rank_count(); ++local_rank) {
    MultiCopyParams copy_out_params;
    const CommRank& comm_rank = comm_group.GetCommRank(local_rank);
    const StreamCtx* stream_ctx = device_id2stream_ctx.at(comm_rank.device_id()).get();
    request_store->ForEachMutRequestEntryForIdsInJob(
        request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          copy_out_params.Add(request_entry->GetRuntimeRequest(local_rank)->recv_buff,
                              stream_ctx->fusion_buffer() + offset_vec.at(i),
                              request_entry->size_in_bytes());
        });
    MluCurrentDeviceGuard guard(comm_rank.device_id());
    MultiCopy(stream_ctx->stream(), copy_out_params);
  }
}

void LaunchAggregatedOps(const CommGroup& comm_group,
                         const std::vector<std::unique_ptr<StreamCtx>>& device_id2stream_ctx,
                         const std::shared_ptr<RequestStore>& request_store,
                         const std::vector<RequestId>& request_ids) {
  for (int32_t local_rank = 0; local_rank < comm_group.local_rank_count(); ++local_rank) {
    const CommRank& comm_rank = comm_group.GetCommRank(local_rank);
    const auto comm = comm_rank.cncl_comm();
    const StreamCtx* stream_ctx = device_id2stream_ctx.at(comm_rank.device_id()).get();
    MluCurrentDeviceGuard guard(comm_rank.device_id());
    request_store->ForEachMutRequestEntryForIdsInJob(request_ids, [&](RequestEntry* request_entry,
                                                                      int32_t i,
                                                                      const RequestId& request_id) {
      const auto& op_desc = request_entry->desc().op_desc();
      const std::shared_ptr<const RuntimeRequestInfo>& runtime_request_info =
          request_entry->GetRuntimeRequest(local_rank);
      const OpType op_type = op_desc.op_type();
      const void* send_buff = runtime_request_info->send_buff;
      void* recv_buff = runtime_request_info->recv_buff;
      const int64_t elem_cnt = request_entry->elem_cnt();
      const cnclDataType_t cncl_data_type = GetCnclDataType(op_desc.data_type());
      const int32_t num_ranks = comm_group.global_rank_count();
      if (op_type == OpType::kOpTypeAllReduce) {
        OF_CNCL_CHECK(cnclAllReduce(send_buff, recv_buff, elem_cnt, cncl_data_type,
                                    GetCnclReduceOp(op_desc.reduce_method()), comm,
                                    stream_ctx->stream()));
      } else if (op_type == OpType::kOpTypeAllGather) {
        CHECK_EQ(elem_cnt % num_ranks, 0);
        OF_CNCL_CHECK(cnclAllGather(send_buff, recv_buff, elem_cnt / num_ranks, cncl_data_type,
                                    comm, stream_ctx->stream()));
      } else if (op_type == OpType::kOpTypeReduceScatter) {
        CHECK_EQ(elem_cnt % num_ranks, 0);
        OF_CNCL_CHECK(cnclReduceScatter(send_buff, recv_buff, elem_cnt / num_ranks, cncl_data_type,
                                        GetCnclReduceOp(op_desc.reduce_method()), comm,
                                        stream_ctx->stream()));
      } else if (op_type == OpType::kOpTypeReduce) {
        OF_CNCL_CHECK(cnclReduce(send_buff, recv_buff, elem_cnt, cncl_data_type,
                                 GetCnclReduceOp(op_desc.reduce_method()), op_desc.root(), comm,
                                 stream_ctx->stream()));
      } else if (op_type == OpType::kOpTypeBroadcast) {
        OF_CNCL_CHECK(cnclBroadcast(send_buff, recv_buff, elem_cnt, cncl_data_type, op_desc.root(),
                                    comm, stream_ctx->stream()));
      } else if (op_type == OpType::kOpTypeAll2All) {
        const int64_t elem_per_rank = elem_cnt / num_ranks;
        const int64_t elem_per_chunk = elem_per_rank / num_ranks;
        const int64_t dtype_size = GetSizeOfDataType(op_desc.data_type());
        const int64_t chunk_size = elem_per_chunk * dtype_size;
        int64_t current_rank = GlobalProcessCtx::Rank();
        OF_MLU_CHECK(cnrtMemcpyAsync(
            reinterpret_cast<void*>(reinterpret_cast<char*>(recv_buff) + current_rank * chunk_size),
            const_cast<void*>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(send_buff)
                                                            + current_rank * chunk_size)),
            chunk_size, stream_ctx->stream(), cnrtMemcpyDevToDev));
        for (int64_t j = 0; j < current_rank; ++j) {
          OF_CNCL_CHECK(
              cnclRecv(reinterpret_cast<void*>(reinterpret_cast<char*>(recv_buff) + j * chunk_size),
                       elem_per_chunk, cncl_data_type, j, comm, stream_ctx->stream()));
        }
        for (int64_t j = current_rank + 1; j < num_ranks; ++j) {
          OF_CNCL_CHECK(cnclSend(const_cast<void*>(reinterpret_cast<const void*>(
                                     reinterpret_cast<const char*>(send_buff) + j * chunk_size)),
                                 elem_per_chunk, cncl_data_type, j, comm, stream_ctx->stream()));
        }
        for (int64_t j = 0; j < current_rank; ++j) {
          OF_CNCL_CHECK(cnclSend(const_cast<void*>(reinterpret_cast<const void*>(
                                     reinterpret_cast<const char*>(send_buff) + j * chunk_size)),
                                 elem_per_chunk, cncl_data_type, j, comm, stream_ctx->stream()));
        }
        for (int64_t j = current_rank + 1; j < num_ranks; ++j) {
          OF_CNCL_CHECK(
              cnclRecv(reinterpret_cast<void*>(reinterpret_cast<char*>(recv_buff) + j * chunk_size),
                       elem_per_chunk, cncl_data_type, j, comm, stream_ctx->stream()));
        }
      } else {
        UNIMPLEMENTED();
      }
    });
  }
}

void AddCallbackAndResetRuntimeRequest(
    const CommGroup& comm_group,
    const std::vector<std::unique_ptr<StreamCtx>>& device_id2stream_ctx,
    const std::shared_ptr<RequestStore>& request_store, const std::vector<RequestId>& request_ids) {
  std::vector<std::vector<std::shared_ptr<const RuntimeRequestInfo>>> saved_runtime_request_info(
      request_ids.size());
  request_store->ForEachMutRequestEntryForIdsInJob(
      request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
        saved_runtime_request_info.at(i) = std::move(request_entry->ResetRuntimeRequest());
      });
  for (int32_t local_rank = 0; local_rank < comm_group.local_rank_count(); ++local_rank) {
    const CommRank& comm_rank = comm_group.GetCommRank(local_rank);
    StreamCtx* stream_ctx = device_id2stream_ctx.at(comm_rank.device_id()).get();
    auto runtime_request_info_vec =
        std::make_shared<std::vector<std::shared_ptr<const RuntimeRequestInfo>>>();
    runtime_request_info_vec->reserve(request_ids.size());
    request_store->ForEachMutRequestEntryForIdsInJob(
        request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          runtime_request_info_vec->emplace_back(
              std::move(saved_runtime_request_info.at(i).at(local_rank)));
        });
    MluCurrentDeviceGuard guard(comm_rank.device_id());
    stream_ctx->AddCallback([runtime_request_info_vec]() {
      for (auto& runtime_request_info : *runtime_request_info_vec) {
        runtime_request_info->callback(Maybe<void>::Ok());
      }
    });
  }
}

}  // namespace

class CnclExecutorBackend : public ExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CnclExecutorBackend);
  CnclExecutorBackend();
  ~CnclExecutorBackend() override;

 private:
  void Init(std::shared_ptr<RequestStore> request_store) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;
  void GroupRequests(const std::vector<RequestId>& request_ids,
                     const std::function<void(std::vector<RequestId>&&, void*)>& Handler) override;
  void ExecuteGroup(void* group_token) override;
  void* CreateGroupToken(const std::vector<RequestId>& group) override;
  void DestroyGroupToken(void* group_token) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CnclExecutorBackend::Impl {
  Impl(const CollectiveBoxingConf& conf, std::shared_ptr<RequestStore> request_store)
      : conf(conf), request_store(std::move(request_store)) {
    CHECK_GT(conf.nccl_num_streams(), 0);
    CHECK_GE(conf.nccl_fusion_threshold_mb(), 0);
    fusion_threshold = conf.nccl_fusion_threshold_mb() * 1024 * 1024;
    num_streams = conf.nccl_num_streams();
    current_stream_id = 0;
    enable_mixed_fusion =
        (!conf.nccl_fusion_all_reduce_use_buffer()) && conf.nccl_enable_mixed_fusion();
    InitStreamCtx();
    InitIsOpTypeFusionEnabled();
  }
  ~Impl() {
    stream_id2device_id2stream_ctx.clear();
    device_set2stream_id2comm_group.clear();
  }

  void InitCommGroup(int64_t job_id) {
    std::set<int64_t> local_device_ids;
    request_store->ForEachMutRequestEntryInJob(
        job_id, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          const auto& request = request_entry->desc();
          if (request.op_desc().device_type() != DeviceType::kMLU) { return; }
          if (!request_entry->HasRankOnThisNode()) { return; }
          const DeviceSet& device_set = request.device_set();
          if (device_set2stream_id2comm_group.count(device_set) > 0) { return; }
          auto& stream_id2comm_group = device_set2stream_id2comm_group[device_set];
          stream_id2comm_group.resize(num_streams);
          for (int32_t stream_id = 0; stream_id < num_streams; ++stream_id) {
            stream_id2comm_group.at(stream_id).InitGroup(
                device_set, GetCnclUniqueIdRpcKey(request.op_desc().name(), stream_id));
          }
          for (int32_t j = 0; j < stream_id2comm_group.at(0).local_rank_count(); ++j) {
            local_device_ids.emplace(stream_id2comm_group.at(0).GetCommRank(j).device_id());
          }
        });
    for (int32_t stream_id = 0; stream_id < num_streams; ++stream_id) {
      for (const int64_t device_id : local_device_ids) {
        if (stream_id2device_id2stream_ctx.at(stream_id).at(device_id) == nullptr) {
          stream_id2device_id2stream_ctx.at(stream_id).at(device_id) =
              std::make_unique<StreamCtx>(device_id, fusion_threshold);
        }
      }
    }
  }

  void InitStreamCtx() {
    uint32_t num_devices = 1;
    OF_MLU_CHECK(cnrtGetDeviceCount(&num_devices));
    stream_id2device_id2stream_ctx.resize(num_streams);
    for (int64_t stream_id = 0; stream_id < num_streams; ++stream_id) {
      stream_id2device_id2stream_ctx.at(stream_id).resize(num_devices);
    }
  }

  void InitIsOpTypeFusionEnabled() {
    op_type2fusion_enabled.resize(OpType_ARRAYSIZE, false);
    op_type2fusion_enabled.at(OpType::kOpTypeAllReduce) = conf.nccl_fusion_all_reduce();
    op_type2fusion_enabled.at(OpType::kOpTypeAllGather) = conf.nccl_fusion_all_gather();
    op_type2fusion_enabled.at(OpType::kOpTypeReduceScatter) = conf.nccl_fusion_reduce_scatter();
    op_type2fusion_enabled.at(OpType::kOpTypeReduce) = conf.nccl_fusion_reduce();
    op_type2fusion_enabled.at(OpType::kOpTypeBroadcast) = conf.nccl_fusion_broadcast();
    op_type2fusion_enabled.at(OpType::kOpTypeAll2All) = false;
  }

  int32_t NextStreamId() {
    const int32_t stream_id = current_stream_id;
    current_stream_id = (current_stream_id + 1) % num_streams;
    return stream_id;
  }

  bool IsOpTypeFusionEnabled(OpType op_type) const { return op_type2fusion_enabled.at(op_type); }

  bool IsRequestEntryFusionEnabled(const RequestEntry* entry) const {
    return IsOpTypeFusionEnabled(entry->desc().op_desc().op_type());
  }

  bool CanRequestEntryFuse(const RequestEntry* lhs, const RequestEntry* rhs) const {
    if (lhs->device_set_symbol() != rhs->device_set_symbol()) { return false; }
    if ((!IsRequestEntryFusionEnabled(lhs)) || (!IsRequestEntryFusionEnabled(rhs))) {
      return false;
    }
    if ((!enable_mixed_fusion)
        && lhs->desc().op_desc().op_type() != rhs->desc().op_desc().op_type()) {
      return false;
    }
    if (conf.nccl_fusion_all_reduce_use_buffer()) {
      if (lhs->desc().op_desc().op_type() == OpType::kOpTypeAllReduce
          && rhs->desc().op_desc().op_type() == OpType::kOpTypeAllReduce) {
        CHECK(lhs->desc().op_desc().has_reduce_method());
        CHECK(rhs->desc().op_desc().has_reduce_method());
        return lhs->desc().op_desc().reduce_method() == rhs->desc().op_desc().reduce_method()
               && lhs->desc().op_desc().data_type() == rhs->desc().op_desc().data_type();
      } else if (lhs->desc().op_desc().op_type() == OpType::kOpTypeAllReduce
                 || rhs->desc().op_desc().op_type() == OpType::kOpTypeAllReduce) {
        return false;
      } else {
        return true;
      }
    } else {
      return true;
    }
  }

  void GroupRequests(const std::vector<RequestId>& request_ids,
                     const std::function<void(std::vector<RequestId>&&, void*)>& Handler) {
    std::vector<RequestId> group;
    int64_t group_size = 0;
    const int64_t fusion_max_ops = std::min(conf.nccl_fusion_max_ops(), kMultiCopyParamsMaxSize);
    request_store->ForEachMutRequestEntryForIdsInJob(
        request_ids, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          const auto& request = request_entry->desc();
          const int64_t size = GetMultiCopyAlignedSize(request_entry->size_in_bytes());
          if (group.empty()
              || !CanRequestEntryFuse(request_store->MutRequestEntry(group.back()), request_entry)
              || group_size + size > fusion_threshold || group.size() >= fusion_max_ops) {
            if (!group.empty()) {
              void* token = CreateGroupToken(group);
              Handler(std::move(group), token);
              group.clear();
              group_size = 0;
            }
          }
          group.emplace_back(request_id);
          group_size += size;
        });
    if (!group.empty()) {
      void* token = CreateGroupToken(group);
      Handler(std::move(group), token);
    }
  }

  struct GroupToken {
    GroupToken(const std::vector<RequestId>& group, std::vector<CommGroup>* stream_id2comm_group)
        : request_ids(group), stream_id2comm_group(stream_id2comm_group) {}
    std::vector<RequestId> request_ids;
    std::vector<CommGroup>* stream_id2comm_group;
  };

  void* CreateGroupToken(const std::vector<RequestId>& group) {
    CHECK_GT(group.size(), 0);
    void* group_token;
    const DeviceSet& first_device_set =
        request_store->MutRequestEntry(group.front())->desc().device_set();
    auto it = device_set2stream_id2comm_group.find(first_device_set);
    CHECK(it != device_set2stream_id2comm_group.end());
    group_token = new GroupToken(group, &it->second);
    request_store->ForEachMutRequestEntryForIdsInJob(
        group, [&](RequestEntry* request_entry, int32_t i, const RequestId& request_id) {
          const DeviceSet& device_set = request_entry->desc().device_set();
          CHECK(first_device_set == device_set);
        });
    return group_token;
  }

  void DestroyGroupToken(void* group_token) {
    GroupToken* token = static_cast<GroupToken*>(group_token);
    delete token;
  }

  void ExecuteGroup(void* group_token) {
    GroupToken* token = static_cast<GroupToken*>(group_token);
    const std::vector<RequestId>& request_ids = token->request_ids;
    if (request_ids.empty()) { return; }
    const int32_t stream_id = NextStreamId();
    const auto& comm_group = token->stream_id2comm_group->at(stream_id);
    auto& device_id2stream_ctx = stream_id2device_id2stream_ctx.at(stream_id);
    if (request_store->MutRequestEntry(request_ids.front())->desc().op_desc().op_type()
            == OpType::kOpTypeAllReduce
        && conf.nccl_fusion_all_reduce_use_buffer() && request_ids.size() > 1) {
      LaunchFusedAllReduce(comm_group, device_id2stream_ctx, request_store, request_ids);
    } else {
      LaunchAggregatedOps(comm_group, device_id2stream_ctx, request_store, request_ids);
    }
    AddCallbackAndResetRuntimeRequest(comm_group, device_id2stream_ctx, request_store, request_ids);
  }

  CollectiveBoxingConf conf;
  int64_t fusion_threshold;
  int32_t num_streams;
  int32_t current_stream_id;
  bool enable_mixed_fusion;
  std::vector<bool> op_type2fusion_enabled;
  std::shared_ptr<RequestStore> request_store;
  HashMap<DeviceSet, std::vector<CommGroup>> device_set2stream_id2comm_group;
  std::vector<std::vector<std::unique_ptr<StreamCtx>>> stream_id2device_id2stream_ctx;
};

CnclExecutorBackend::CnclExecutorBackend() = default;

CnclExecutorBackend::~CnclExecutorBackend() = default;

void CnclExecutorBackend::Init(std::shared_ptr<RequestStore> request_store) {
  impl_ = std::make_unique<Impl>(
      Singleton<ResourceDesc, ForSession>::Get()->collective_boxing_conf(), request_store);
}

void CnclExecutorBackend::InitJob(int64_t job_id) { impl_->InitCommGroup(job_id); }

void CnclExecutorBackend::DeinitJob(int64_t job_id) {}

void CnclExecutorBackend::GroupRequests(
    const std::vector<RequestId>& request_ids,
    const std::function<void(std::vector<RequestId>&&, void*)>& Handler) {
  impl_->GroupRequests(request_ids, Handler);
}

void* CnclExecutorBackend::CreateGroupToken(const std::vector<RequestId>& group) {
  return impl_->CreateGroupToken(group);
}

void CnclExecutorBackend::DestroyGroupToken(void* group_token) {
  return impl_->DestroyGroupToken(group_token);
}

void CnclExecutorBackend::ExecuteGroup(void* group_token) { impl_->ExecuteGroup(group_token); }

REGISTER_EXECUTOR_BACKEND(DeviceType::kMLU, CnclExecutorBackend);

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
