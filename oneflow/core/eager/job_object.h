#ifndef ONEFLOW_CORE_EAGER_JOB_OBJECT_H_
#define ONEFLOW_CORE_EAGER_JOB_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace eager {

class JobObject : public vm::Object {
 public:
  JobObject(const JobObject&) = delete;
  JobObject(JobObject&&) = delete;

  JobObject(const std::shared_ptr<Job>& job, int64_t job_id)
      : job_(job), job_desc_(job->job_conf(), job_id) {
    Init();
  }
  ~JobObject() override = default;

  const JobDesc& job_desc() const { return job_desc_; }
  const Job& job() const { return *job_; }
  bool HasOpConf(int64_t op_logical_object_id) const;
  const OperatorConf& OpConf4LogicalObjectId(int64_t op_logical_object_id) const;
  int64_t LogicalObjectId4Lbi(const LogicalBlobId& lbi) const;
  const ParallelDesc& parallel_desc() const { return *parallel_desc_; }

 private:
  void Init() {
    InitLogicalObjectId2OpConf();
    InitLbi2LogicalObjectId();
    InitParallelDesc();
  }
  void InitLogicalObjectId2OpConf();
  void InitLbi2LogicalObjectId();
  void InitParallelDesc();

  std::shared_ptr<Job> job_;
  const JobDesc job_desc_;
  HashMap<int64_t, const OperatorConf*> logical_object_id2op_conf_;
  HashMap<LogicalBlobId, int64_t> lbi2logical_object_id_;
  std::shared_ptr<ParallelDesc> parallel_desc_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_JOB_OBJECT_H_
