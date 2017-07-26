#include "oneflow/core/distributed_runtime/master.h"

namespace oneflow {

Master::Master() {}

Master::~Master() {}

::tensorflow::Status Master::SendJob(SendJobRequest* request,
                                     SendJobResponse* response,
                                     MyClosure done) {
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
