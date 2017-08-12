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
  //JobDescProto job_desc;
  //JobConf job_conf;
  //job_conf.CopyFrom(request->job_conf());
  //DLNetConf net_conf;
  //net_conf.CopyFrom(request->dlnet_conf());
  //Resource resource_conf;
  //resource_conf.CopyFrom(request->resource_conf());
  //Placement placement_conf;
  //placement_conf.CopyFrom(request->placement_conf());

  //job_desc.set_allocated_job_conf(&job_conf);
  //job_desc.set_allocated_train_dlnet_conf(&net_conf);
  //job_desc.set_allocated_resource(&resource_conf);
  //job_desc.set_allocated_placement(&placement_conf);

  //::oneflow::compiler::Compiler::Singleton()->Compile(job_desc,
  //                                                    response->mutable_plan());

  std::string str_request;
  PrintProtoToString(*request, &str_request);
  LOG(INFO) << str_request;

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
