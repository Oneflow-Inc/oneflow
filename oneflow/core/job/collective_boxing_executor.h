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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

namespace boxing {

namespace collective {

struct RuntimeRequestInfo {
  const void* send_buff;
  void* recv_buff;
  std::shared_ptr<const std::function<void(const Maybe<void>&)>> callback;
};

class CollectiveBoxingExecutorBackend {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingExecutorBackend);
  CollectiveBoxingExecutorBackend() = default;
  virtual ~CollectiveBoxingExecutorBackend() = default;

  virtual void Init(const CollectiveBoxingPlan& collective_boxing_plan){};
  virtual void GroupRequests(const std::vector<const RequestDesc*>& requests,
                             std::vector<std::vector<const RequestDesc*>>* groups);
  virtual void ExecuteGroup(const std::vector<const RequestDesc*>& group,
                            const std::vector<std::map<int64_t, RuntimeRequestInfo>>& ranks) = 0;
};

class CollectiveBoxingExecutor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingExecutor);
  ~CollectiveBoxingExecutor() = default;

  void Enqueue(const RankDesc& rank_desc, const RuntimeRequestInfo& request_info);

 private:
  friend class Global<CollectiveBoxingExecutor>;
  explicit CollectiveBoxingExecutor(const Plan& plan);

  void Init();
  void DumpSummary() const;

  struct RequestState {
    RequestState(const RequestDesc* p_request_desc, int64_t p_job_id, int64_t p_group_id,
                 std::set<int64_t> p_local_ranks)
        : request_desc(p_request_desc),
          job_id(p_job_id),
          group_id(p_group_id),
          local_ranks(std::move(p_local_ranks)),
          ready_ranks() {}
    const RequestDesc* const request_desc;
    const int64_t job_id;
    const int64_t group_id;
    const std::set<int64_t> local_ranks;
    std::map<int64_t, RuntimeRequestInfo> ready_ranks;

    void AddReadyRank(const RankDesc& rank_desc, const RuntimeRequestInfo& request_info);
    bool IsReady() const;
  };

  struct GroupState {
    GroupState(CollectiveBoxingExecutorBackend* p_backend, std::set<int64_t> p_request_ids,
               std::vector<const RequestDesc*> p_requests)
        : backend(p_backend),
          request_ids(std::move(p_request_ids)),
          requests(std::move(p_requests)),
          ready_request_ids() {}
    CollectiveBoxingExecutorBackend* const backend;
    const std::set<int64_t> request_ids;
    const std::vector<const RequestDesc*> requests;
    std::set<int64_t> ready_request_ids;

    void AddReadyRequest(int64_t request_id);
    bool IsReady() const;
  };

  std::mutex mutex_;

  const CollectiveBoxingPlan collective_boxing_plan_;
  std::map<Backend, std::unique_ptr<CollectiveBoxingExecutorBackend>> backends_;
  HashMap<std::string, int64_t> name2request_id_;
  std::vector<RequestState> request_id2request_state_;
  std::map<int64_t, std::vector<int64_t>> job_id2group_ids_;
  std::vector<GroupState> group_id2group_state_;

  int64_t current_job_id_ = -1;
  int64_t current_group_idx_in_job_ = -1;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
