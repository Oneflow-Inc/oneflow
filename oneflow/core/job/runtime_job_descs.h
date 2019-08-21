#ifndef ONEFLOW_CORE_JOB_RUNTIME_JOB_DESCS_H_
#define ONEFLOW_CORE_JOB_RUNTIME_JOB_DESCS_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class RuntimeJobDescs final {
 public:
  explicit RuntimeJobDescs(const PbMap<int64_t, JobConfigProto>& proto);
  const JobDesc& job_desc(int64_t job_id) const { return *job_id2job_desc_.at(job_id); }

 private:
  HashMap<int64_t, std::unique_ptr<JobDesc>> job_id2job_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_JOB_DESCS_H_
