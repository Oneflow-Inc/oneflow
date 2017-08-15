#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_

#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Worker {
 public:
  Worker(const std::string& this_node_name);
  ~Worker();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendPlan(SendPlanRequest* request,
                                SendPlanResponse* response, MyClosure done);

 private:
  std::string this_node_name_;
  Plan plan_;
};  // Worker
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
