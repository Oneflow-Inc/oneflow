#include "oneflow/core/eager/job_storage.h"

namespace oneflow {
namespace eager {

bool JobStorage::HasJob(int64_t logical_object_id) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return logical_object_id2job_.find(logical_object_id) != logical_object_id2job_.end();
}

std::shared_ptr<Job> JobStorage::LookupJob(int64_t logical_object_id) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return logical_object_id2job_.at(logical_object_id);
}

void JobStorage::AddJob(const std::shared_ptr<Job>& job) {
  CHECK(job->job_conf().has_logical_object_id());
  int64_t logical_object_id = job->job_conf().logical_object_id();
  CHECK_GT(logical_object_id, 0);
  std::unique_lock<std::mutex> lock(mutex_);
  CHECK(logical_object_id2job_.emplace(logical_object_id, job).second);
}

void JobStorage::ClearJob(int64_t logical_object_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  logical_object_id2job_.erase(logical_object_id);
}

}  // namespace eager
}  // namespace oneflow
