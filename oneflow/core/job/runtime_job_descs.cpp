#include "oneflow/core/job/runtime_job_descs.h"

namespace oneflow {

RuntimeJobDescs::RuntimeJobDescs(const PbMap<int64_t, JobConfigProto>& proto) {
  for (const auto& pair : proto) {
    auto job_desc = std::make_unique<JobDesc>(pair.second, pair.first);
    CHECK(job_id2job_desc_.emplace(pair.first, std::move(job_desc)).second);
  }
}

}  // namespace oneflow
