#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unordered_map>

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/resource.pb.h"

#include <grpc++/grpc++.h>

#include "oneflow/core/distributed_runtime/rpc/grpc_remote_master.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"

DEFINE_string(master_addr, "", "");
DEFINE_string(job_conf_filepath, "", "");

int main(int argc, char** argv) {
  using namespace oneflow;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  JobConf job_conf;
  ParseProtoFromTextFile(FLAGS_job_conf_filepath, &job_conf);
  std::string dlnet_conf_filepath = job_conf.train_dlnet_conf_filepath();
  std::string resource_filepath = job_conf.resource_filepath();
  std::string placement_filepath = job_conf.placement_filepath();

  std::string master_address = FLAGS_master_addr;

  DLNetConf dlnet_conf;
  ParseProtoFromTextFile(dlnet_conf_filepath, &dlnet_conf);

  Resource resource_conf;
  ParseProtoFromTextFile(resource_filepath, &resource_conf);

  Placement placement_conf;
  ParseProtoFromTextFile(placement_filepath, &placement_conf);
  LOG(INFO) << "Starting client";

  ::tensorflow::Status s;

  std::shared_ptr<::grpc::Channel> channel = ::grpc::CreateChannel(
      master_address, ::grpc::InsecureChannelCredentials());

  std::shared_ptr<GrpcRemoteMaster> remote_master(
      new GrpcRemoteMaster(channel));
  SendJobRequest req;
  *(req.mutable_job_conf()) = job_conf;
  *(req.mutable_dlnet_conf()) = dlnet_conf;
  *(req.mutable_resource_conf()) = resource_conf;
  *(req.mutable_placement_conf()) = placement_conf;
  SendJobResponse resp;

  s = remote_master->SendJob(&req, &resp);
  if (s.ok()) {
    LOG(INFO) << "SendJob RPC succeeds";
    std::string str_plan;
    PrintProtoToString(resp.plan(), &str_plan);
    LOG(INFO) << str_plan;
    std::ofstream fout("plan_test");
    fout << str_plan << std::endl;
  } else {
    LOG(INFO) << "SendJob RPC fails";
  }

  MasterInitDataPlaneRequest master_init_dp_req;
  MasterInitDataPlaneResponse master_init_dp_resp;
  s = remote_master->MasterInitDataPlane(&master_init_dp_req,
                                         &master_init_dp_resp);

  if (s.ok()) {
    LOG(INFO) << "MasterInitDataPlane RPC succeeds";
  } else {
    LOG(INFO) << "MasterInitDataPlane RPC fails";
  }

  return 0;
}
