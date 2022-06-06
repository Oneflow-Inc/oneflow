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
#include "oneflow/core/job/of_collective_boxing/collective_backend_ofccl.h"
#include "oneflow/core/job/of_collective_boxing/of_request_store.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/device/cuda_util.h"

#include <nccl.h>

#include <memory>
#include <utility>

namespace oneflow {

namespace boxing {

// TODO: (Panlichen) !!!! UNFINISHED !!!!!
namespace of_collective {

namespace {

ncclRedOp_t GetNcclReduceOp(ReduceMethod reduce_method) {
  if (reduce_method == kReduceMethodSum) {
    return ncclRedOp_t::ncclSum;
  } else {
    UNIMPLEMENTED();
    return ncclRedOp_t{};
  }
}

std::string GetNcclUniqueIdRpcKey(const std::string& name, int64_t stream_id) {
  return "CollectiveBoxingExecutorNcclUniqueIdRpcKey-" + name + "-" + std::to_string(stream_id);
}

struct CopyParams {
  void* dst;
  const void* src;
  int64_t count;
};

constexpr int64_t kMultiCopyParamsMaxSize = 128;
constexpr int64_t kMultiCopyAlignSize = 32;

int64_t GetMultiCopyAlignedSize(int64_t size) {
  return ((size + kMultiCopyAlignSize - 1) / kMultiCopyAlignSize) * kMultiCopyAlignSize;
}

struct MultiCopyParams {
  CopyParams params[kMultiCopyParamsMaxSize];
  int64_t count;

  MultiCopyParams() : count(0), params{} {}

  void Add(void* dst, const void* src, int64_t count) {
    CHECK_LT(this->count, kMultiCopyParamsMaxSize);
    params[this->count].dst = dst;
    params[this->count].src = src;
    params[this->count].count = count;
    this->count += 1;
  }
};

class CommRank final {
 public:
  OF_DISALLOW_COPY(CommRank);
  CommRank(int32_t device_id, int32_t global_rank, int32_t global_rank_count, int32_t local_rank,
           int32_t local_rank_count)
      : device_id_(device_id),
        global_rank_(global_rank),
        local_rank_(local_rank),
        nccl_comm_(nullptr) {}

  CommRank(CommRank&& rhs) noexcept {
    this->device_id_ = rhs.device_id_;
    this->global_rank_ = rhs.global_rank_;
    this->local_rank_ = rhs.local_rank_;
    this->nccl_comm_ = rhs.nccl_comm_;
    rhs.nccl_comm_ = nullptr;
  }

  ~CommRank() {
    if (nccl_comm_ != nullptr) {
      CudaCurrentDeviceGuard guard(device_id_);
      OF_NCCL_CHECK(ncclCommDestroy(nccl_comm_));
    }
  }

  int32_t device_id() const { return device_id_; }

  ncclComm_t nccl_comm() const { return nccl_comm_; }

  void InitRank(ncclUniqueId unique_id, int32_t global_rank_count) {
    CudaCurrentDeviceGuard guard(device_id_);
    OF_NCCL_CHECK(ncclCommInitRank(&nccl_comm_, global_rank_count, unique_id, global_rank_));
  }

 private:
  int32_t device_id_;
  int32_t global_rank_;
  int32_t local_rank_;
  ncclComm_t nccl_comm_;
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
    CudaCurrentDeviceGuard guard;
    const int64_t this_machine_id = GlobalProcessCtx::Rank();
    global_rank_count_ = device_set.device_size();
    std::vector<int32_t> local_ranks;
    for (int32_t i = 0; i < global_rank_count_; ++i) {
      if (device_set.device(i).machine_id() == this_machine_id) { local_ranks.emplace_back(i); }
    }
    const int32_t local_rank_count = local_ranks.size();
    CHECK_GT(local_rank_count, 0);
    ncclUniqueId nccl_unique_id{};
    if (local_ranks.front() == 0) {
      OF_NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id));
      if (local_rank_count != global_rank_count_) {
        Global<CtrlClient>::Get()->PushKV(unique_name, NcclUniqueIdToString(nccl_unique_id));
      }
    } else {
      Global<CtrlClient>::Get()->PullKV(unique_name, [&nccl_unique_id](const std::string& val) {
        NcclUniqueIdFromString(val, &nccl_unique_id);
      });
    }
    rank_vec_.reserve(local_rank_count);
    OF_NCCL_CHECK(ncclGroupStart());
    for (int32_t local_rank = 0; local_rank < local_ranks.size(); ++local_rank) {
      const int32_t global_rank = local_ranks.at(local_rank);
      const int32_t device_id = device_set.device(global_rank).device_id();
      OF_CUDA_CHECK(cudaSetDevice(device_id));
      rank_vec_.emplace_back(device_id, global_rank, global_rank_count_, local_rank,
                             local_rank_count);
      rank_vec_.at(local_rank).InitRank(nccl_unique_id, global_rank_count_);
    }
    OF_NCCL_CHECK(ncclGroupEnd());
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
    CudaCurrentDeviceGuard guard(device_id_);
    int priority;
    OF_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(nullptr, &priority));
    OF_CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, priority));
    OF_CUDA_CHECK(cudaMalloc(&fusion_buffer_, fusion_buffer_size_));
    cb_event_poller_ = std::thread(&StreamCtx::PollEvent, this);
  }
  ~StreamCtx() {
    cb_event_chan_.Close();
    cb_event_poller_.join();
    CudaCurrentDeviceGuard guard(device_id_);
    OF_CUDA_CHECK(cudaStreamSynchronize(stream_));
    OF_CUDA_CHECK(cudaStreamDestroy(stream_));
    OF_CUDA_CHECK(cudaFree(fusion_buffer_));
  }

  void PollEvent() {
    CudaCurrentDeviceGuard guard(device_id_);
    while (true) {
      std::pair<cudaEvent_t, std::function<void()>> cb_event;
      ChannelStatus status = cb_event_chan_.Receive(&cb_event);
      if (status == kChannelStatusErrorClosed) { break; }
      CHECK_EQ(status, kChannelStatusSuccess);
      OF_CUDA_CHECK(cudaEventSynchronize(cb_event.first));
      cb_event.second();
      OF_CUDA_CHECK(cudaEventDestroy(cb_event.first));
    }
  }

  void AddCallback(const std::function<void()>& callback) {
    cudaEvent_t event;
    OF_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    OF_CUDA_CHECK(cudaEventRecord(event, stream_));
    CHECK_EQ(cb_event_chan_.Send(std::make_pair(event, callback)), kChannelStatusSuccess);
  }

  int32_t device_id() const { return device_id_; }

  cudaStream_t stream() const { return stream_; }

  size_t fusion_buffer_size() const { return fusion_buffer_size_; }

  char* fusion_buffer() const { return fusion_buffer_; }

 private:
  int32_t device_id_;
  cudaStream_t stream_ = nullptr;
  size_t fusion_buffer_size_;
  char* fusion_buffer_ = nullptr;
  Channel<std::pair<cudaEvent_t, std::function<void()>>> cb_event_chan_;
  std::thread cb_event_poller_;
};


}  // namespace

struct CollectiveBackendOfccl::Impl {
  Impl(const CollectiveBoxingConf& conf, std::shared_ptr<OfRequestStore> request_store)
      : conf(conf), request_store(std::move(request_store)) {
    CHECK_GT(conf.nccl_num_streams(), 0);
    num_streams = conf.nccl_num_streams();
    current_stream_id = 0;
    int nccl_version;
    OF_NCCL_CHECK(ncclGetVersion(&nccl_version));
    if (nccl_version == 21003) {
      LOG(WARNING)
          << "Current nccl version is 2.10.3, in this version, ncclGroup() with mixed "
             "datatype/element/collective could induce crash or corruption, so we will not "
             "fuse any request.";
    }
    InitStreamCtx();
  }
  ~Impl() {
    stream_id2device_id2stream_ctx.clear();
    device_set2stream_id2comm_group.clear();
  }

  void InitCommGroup(int64_t job_id) {
    return;
  }

  void InitStreamCtx() {
    int32_t num_devices;
    OF_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    stream_id2device_id2stream_ctx.resize(num_streams);
    for (int64_t stream_id = 0; stream_id < num_streams; ++stream_id) {
      stream_id2device_id2stream_ctx.at(stream_id).resize(num_devices);
    }
  }

  int32_t NextStreamId() {
    const int32_t stream_id = current_stream_id;
    current_stream_id = (current_stream_id + 1) % num_streams;
    return stream_id;
  }

  CollectiveBoxingConf conf;
  int32_t num_streams;
  int32_t current_stream_id;
  std::shared_ptr<OfRequestStore> request_store;
  HashMap<DeviceSet, std::vector<CommGroup>> device_set2stream_id2comm_group;
  std::vector<std::vector<std::unique_ptr<StreamCtx>>> stream_id2device_id2stream_ctx;
};

CollectiveBackendOfccl::CollectiveBackendOfccl() = default;

CollectiveBackendOfccl::~CollectiveBackendOfccl() = default;

void CollectiveBackendOfccl::Init(std::shared_ptr<OfRequestStore> request_store) {
  impl_ = std::make_unique<Impl>(Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf(),
                                 request_store);
}

void CollectiveBackendOfccl::InitJob(int64_t job_id) {
  CudaCurrentDeviceGuard guard;
  impl_->InitCommGroup(job_id);
}

void CollectiveBackendOfccl::DeinitJob(int64_t job_id) {}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
