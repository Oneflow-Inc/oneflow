#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_

#include "oneflow/core/distributed_runtime/master.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Master {
 public:
  Master();
  ~Master();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendJob(SendJobRequest* request,
                               SendJobResponse* response, MyClosure done);

 private:
};  // Master
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
