#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_

#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Worker {
 public:
  Worker();
  ~Worker();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendPlan(SendPlanRequest* request,
                                SendPlanResponse* response, MyClosure done);

 private:
};  // Worker
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
