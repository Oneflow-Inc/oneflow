#ifndef ONEFLOW_CORE_EAGER_JOB_STORAGE_H_
#define ONEFLOW_CORE_EAGER_JOB_STORAGE_H_

#include <mutex>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {
namespace eager {

class JobStorage final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobStorage);

  JobStorage() = default;
  ~JobStorage() = default;

  bool HasJob(int64_t logical_object_id) const;
  const Job& LookupJob(int64_t logical_object_id) const;

  void AddJob(const std::shared_ptr<Job>& job);
  void ClearJob(int64_t logical_object_id);

 private:
  mutable std::mutex mutex_;
  HashMap<int64_t, std::shared_ptr<Job>> logical_object_id2job_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_JOB_STORAGE_H_
