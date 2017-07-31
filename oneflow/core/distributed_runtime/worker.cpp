#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"

namespace oneflow {

Worker::Worker() {}

Worker::~Worker() {}

::tensorflow::Status Worker::SendPlan(SendPlanRequest* request,
                                      SendPlanResponse* response,
                                      MyClosure done) {
  std::string str_request;
  PrintProtoToString(*request, &str_request);
  LOG(INFO) << str_request;
  // PrintProtoToTextFile(request->plan(), "tmp_plan2");
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
