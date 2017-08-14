#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common_runtime/runtime.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"

namespace oneflow {

Worker::Worker(const std::string& this_node_name)
    : this_node_name_(this_node_name) {}

Worker::~Worker() {}

::tensorflow::Status Worker::SendPlan(SendPlanRequest* request,
                                      SendPlanResponse* response,
                                      MyClosure done) {
  std::string str_plan;
  PrintProtoToString(request->plan(), &str_plan);
  LOG(INFO) << str_plan;

  // Plan plan = request->plan();
  // oneflow::runtime::Runtime::Singleton()->Run(plan, this_node_name_);

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
