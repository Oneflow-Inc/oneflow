#ifndef ONEFLOW_CORE_JOB_ONEFLOW_H_
#define ONEFLOW_CORE_JOB_ONEFLOW_H_

#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/control/ctrl_server.h"
#include "oneflow/core/job/runtime.h"

namespace oneflow {

class GlobalObjectsScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GlobalObjectsScope);
  GlobalObjectsScope(const JobSet& job_set);
  ~GlobalObjectsScope();

 private:
  std::unique_ptr<CtrlServer> ctrl_server_;
};

class RuntimeBuffersScope final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeBuffersScope);
  RuntimeBuffersScope();
  ~RuntimeBuffersScope();
};

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  Oneflow(const oneflow::JobSet& job_set);
  ~Oneflow();

  void NaiveSequentialRun() const;

 private:
  std::unique_ptr<GlobalObjectsScope> global_objects_scope_;
  Plan plan_;
  std::unique_ptr<Runtime> runtime_;
  std::unique_ptr<RuntimeBuffersScope> runtime_buffers_scope_;
};

}  // namespace oneflow

int Main(const oneflow::JobSet& job_set);

#endif  // ONEFLOW_CORE_JOB_ONEFLOW_H_
