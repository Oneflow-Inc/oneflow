#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/compiler/compiler.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

Master::Master() {}

Master::~Master() {}

::tensorflow::Status Master::SendJob(SendJobRequest* request,
                                     SendJobResponse* response,
                                     MyClosure done) {
  JobDescProto job_desc;

  *(job_desc.mutable_job_conf()) = request->job_conf();
  *(job_desc.mutable_train_dlnet_conf()) = request->dlnet_conf();
  *(job_desc.mutable_resource()) = request->resource_conf();
  *(job_desc.mutable_placement()) = request->placement_conf();

  ::oneflow::compiler::Compiler::Singleton()->Compile(job_desc,
                                                      response->mutable_plan());

  //std::string str_plan;
  //PrintProtoToString(response->plan(), &str_plan);
  //LOG(INFO) << str_plan;

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
