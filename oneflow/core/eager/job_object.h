#ifndef ONEFLOW_CORE_EAGER_JOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_JOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

class JobObject : public vm::Object {
 public:
  JobObject(const JobObject&) = delete;
  JobObject(JobObject&&) = delete;

  JobObject(const std::shared_ptr<Job>& job, int64_t job_id)
      : job_(job), job_desc_(job->job_conf(), job_id) {
    InitLogicalObjectId2OpConf();
  }
  ~JobObject() override = default;

  const JobDesc& job_desc() const { return job_desc_; }
  const Job& job() const { return *job_; }
  bool HasOpConf(int64_t op_logical_object_id) const;
  const OperatorConf& LookupOpConf(int64_t op_logical_object_id) const;

 private:
  void InitLogicalObjectId2OpConf();

  std::shared_ptr<Job> job_;
  const JobDesc job_desc_;
  HashMap<int64_t, const OperatorConf*> logical_object_id2op_conf_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_JOB_OBJECT_H_
