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
#include "oneflow/core/job/collective_boxing/dynamic_coordinator.h"
#include "oneflow/core/job/collective_boxing/executor.h"
#include "oneflow/core/job/collective_boxing/request_store.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

#ifdef WITH_MPI

#include <mpi.h>

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

class BitVector final {
 public:
  constexpr static size_t kStoreBits = (sizeof(uint64_t) * 8);
  explicit BitVector(size_t size) : size_(size) {
    const size_t store_size = (size + kStoreBits - 1) / kStoreBits;
    store_.resize(store_size);
  }

  inline bool Test(size_t index) {
    CHECK_LT(index, size_);
    const size_t store_index = index / kStoreBits;
    const size_t bit_idx = index % kStoreBits;
    const uint64_t mask = (static_cast<uint64_t>(1) << bit_idx);
    const uint64_t set = (store_[store_index] & mask);
    return set;
  }

  inline bool Set(size_t index) { return SetValue(index, true); }

  inline bool Clear(size_t index) { return SetValue(index, false); }

  inline size_t Size() const { return size_; }

  inline size_t StoreSize() const { return store_.size(); }

  uint64_t* StorePtr() { return store_.data(); }

 private:
  inline bool SetValue(size_t index, bool value) {
    CHECK_LT(index, size_);
    const size_t store_index = index / kStoreBits;
    const size_t bit_idx = index % kStoreBits;
    const uint64_t mask = (static_cast<uint64_t>(1) << bit_idx);
    const bool old_value = static_cast<bool>(store_[store_index] & mask);
    if (value) {
      store_[store_index] |= mask;
    } else {
      store_[store_index] &= (~mask);
    }
    return old_value;
  }

  size_t size_;
  std::vector<uint64_t> store_;
};

}  // namespace

struct DynamicCoordinator::Impl {
  Impl(std::shared_ptr<RequestStore> request_store, std::shared_ptr<Executor> executor);
  ~Impl();

  void AddRequest(int32_t request_id);
  void ExecuteRequests(const std::vector<int32_t>& request_ids);
  void CoordinatingLoop();

  std::shared_ptr<RequestStore> request_store;
  std::shared_ptr<Executor> executor;
  std::mutex executor_mutex;
  std::vector<int32_t> pending_requests;
  std::mutex pending_requests_mutex;
  std::thread coordinating_thread;
  bool shutdown;
};

DynamicCoordinator::Impl::Impl(std::shared_ptr<RequestStore> request_store,
                               std::shared_ptr<Executor> executor)
    : request_store(std::move(request_store)), executor(std::move(executor)), shutdown(false) {
  coordinating_thread = std::thread(&DynamicCoordinator::Impl::CoordinatingLoop, this);
}

DynamicCoordinator::Impl::~Impl() {
  {
    std::lock_guard<std::mutex> lock(pending_requests_mutex);
    shutdown = true;
  }
  coordinating_thread.join();
}

void DynamicCoordinator::Impl::AddRequest(int32_t request_id) {
  RequestEntry* request_entry = request_store->MutRequestEntry(request_id);
  if (request_entry->NodeCount() == 1) {
    ExecuteRequests(std::vector<int32_t>({request_id}));
  } else {
    std::lock_guard<std::mutex> lock(pending_requests_mutex);
    pending_requests.push_back(request_id);
  }
}

void DynamicCoordinator::Impl::ExecuteRequests(const std::vector<int32_t>& request_ids) {
  std::lock_guard<std::mutex> lock(executor_mutex);
  executor->ExecuteRequests(std::vector<int32_t>(request_ids));
}

void DynamicCoordinator::Impl::CoordinatingLoop() {
  int inited;
  MPI_Initialized(&inited);
  CHECK(inited);
  MPI_Comm mpi_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  const size_t max_multi_node_request_id = request_store->MaxMultiNodeRequestId();
  const size_t bit_vec_size = max_multi_node_request_id + 1;
  const size_t shutdown_index = max_multi_node_request_id;
  BitVector local_ready_vec(bit_vec_size);
  BitVector global_ready_vec(bit_vec_size);
  int64_t pending = 0;
  std::vector<int32_t> ready_request_ids;
  for (int i = 0; i < max_multi_node_request_id; ++i) {
    if (!request_store->MutRequestEntry(i)->HasRankOnThisNode()) { local_ready_vec.Set(i); }
  }
  const CollectiveBoxingConf& conf =
      Global<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  const auto cycle_time_us =
      static_cast<int64_t>(conf.dynamic_coordinator_conf().cycle_time_ms() * 1000);
  const std::chrono::microseconds cycle_time(cycle_time_us);
  auto last_loop_time = std::chrono::system_clock::now();
  while (true) {
    const auto now = std::chrono::system_clock::now();
    const auto duration_to_last_loop = now - last_loop_time;
    last_loop_time = now;
    const auto duration_to_sleep = cycle_time - duration_to_last_loop;
    if (duration_to_sleep > std::chrono::microseconds::zero()) {
      std::this_thread::sleep_for(duration_to_sleep);
    }
    {
      std::lock_guard<std::mutex> lock(pending_requests_mutex);
      for (const auto request_id : pending_requests) {
        CHECK_LT(request_id, max_multi_node_request_id);
        CHECK(!local_ready_vec.Set(request_id));
        pending += 1;
      }
      pending_requests.clear();
      if (shutdown) { local_ready_vec.Set(shutdown_index); }
    }

    MPI_Allreduce(local_ready_vec.StorePtr(), global_ready_vec.StorePtr(),
                  local_ready_vec.StoreSize(), MPI_UNSIGNED_LONG_LONG, MPI_BAND, mpi_comm);

    for (int32_t i = 0; i < max_multi_node_request_id; ++i) {
      if (global_ready_vec.Test(i) && request_store->MutRequestEntry(i)->HasRankOnThisNode()) {
        ready_request_ids.push_back(i);
        CHECK(local_ready_vec.Clear(i));
        pending -= 1;
      }
    }
    if (!ready_request_ids.empty()) {
      ExecuteRequests(std::vector<int32_t>(ready_request_ids));
      ready_request_ids.clear();
    }
    if (global_ready_vec.Test(shutdown_index)) {
      CHECK_EQ(pending, 0);
      break;
    }
  }
  MPI_Comm_free(&mpi_comm);
}

DynamicCoordinator::DynamicCoordinator() = default;

DynamicCoordinator::~DynamicCoordinator() = default;

void DynamicCoordinator::Init(const CollectiveBoxingPlan& collective_boxing_plan,
                              std::shared_ptr<RequestStore> request_store,
                              std::shared_ptr<Executor> executor) {
  impl_.reset(new Impl(request_store, executor));
}

void DynamicCoordinator::AddRequest(int32_t request_id) { impl_->AddRequest(request_id); }

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // WITH_MPI
