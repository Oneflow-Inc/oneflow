#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/distributed_runtime/master.pb.h"

namespace oneflow {

Master::Master() {}

Master::~Master() {}

::tensorflow::Status Master::SendJob(SendJobRequest* request,
                                     SendJobResponse* response,
                                     MyClosure done) {
  std::string str_request;
  PrintProtoToString(*request, &str_request);
  LOG(INFO) << str_request;
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
