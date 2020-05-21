#ifndef ONEFLOW_CORE_JOB_ONEFLOW_H_
#define ONEFLOW_CORE_JOB_ONEFLOW_H_

#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/runtime_buffers_scope.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"

namespace oneflow {

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  Oneflow(const oneflow::JobSet& job_set);
  ~Oneflow();

 private:
  Plan plan_;
  std::unique_ptr<RuntimeBuffersScope> runtime_buffers_scope_;
  std::unique_ptr<Runtime> runtime_;
};

int Main(const oneflow::JobSet& job_set);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ONEFLOW_H_
