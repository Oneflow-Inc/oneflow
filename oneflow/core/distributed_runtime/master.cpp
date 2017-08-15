#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/compiler/compiler.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

Master::Master(const ServerDef& server_def,
               ::grpc::CompletionQueue* completion_queue)
    : server_def_(server_def), cq_(completion_queue) {
  ParseServerDef();
  CreateWorkerCache();
}

void Master::ParseServerDef() {
  this_node_name_ = server_def_.this_node_name();

  int32_t node_num = server_def_.cluster_def().cluster_node_size();
  for (int32_t i = 0; i < node_num; ++i) {
    std::string node_name =
        server_def_.cluster_def().cluster_node(i).node_name();
    ClusterNode cluster_node = server_def_.cluster_def().cluster_node(i);
    CHECK(
        name2node_def_.insert(std::make_pair(node_name, cluster_node)).second);
  }
}

void Master::CreateWorkerCache() {
  for (auto& pair : name2node_def_) {
    auto& name = pair.first;
    auto node_def_it = name2node_def_.find(name);
    CHECK(node_def_it != name2node_def_.end());
    auto& node_def = node_def_it->second;
    auto& ctrl_addr = node_def.ctrl_plane_addr();
    std::string worker_addr = ctrl_addr.addr() + ":" + ctrl_addr.port();
    std::shared_ptr<::grpc::Channel> worker_channel = ::grpc::CreateChannel(
        worker_addr, ::grpc::InsecureChannelCredentials());
    std::shared_ptr<GrpcRemoteWorker> remote_worker(
        new GrpcRemoteWorker(worker_channel, cq_));
    CHECK(name2worker_.insert(std::make_pair(name, remote_worker)).second);
  }
}

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

  // std::string str_plan;
  // PrintProtoToString(response->plan(), &str_plan);
  // LOG(INFO) << str_plan;

  // for (auto& pair : name2worker_) {
  //  struct Call {
  //    SendPlanRequest plan_req;
  //    SendPlanResponse plan_resp;
  //  };
  //  Call* call = new Call;
  //  *(call->plan_req.mutable_plan()) = response->plan();

  //  auto cb = [call](const ::tensorflow::Status& s) {
  //    if (s.ok()) {
  //      LOG(INFO) << "SendPlan RPC succeeds";
  //    } else {
  //      LOG(INFO) << "SendPlan RPC fails";
  //    }
  //    delete call;
  //  };
  //  pair.second->SendPlanAsync(&call->plan_req, &call->plan_resp, cb);
  //}

  for (auto& pair : name2worker_) {
    SendPlanRequest plan_req;
    SendPlanResponse plan_resp;
    *(plan_req.mutable_plan()) = response->plan();
    ::tensorflow::Status s = pair.second->SendPlan(&plan_req, &plan_resp);
    CHECK(s.ok());
  }

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterInitDataPlane(
    MasterInitDataPlaneRequest* request, MasterInitDataPlaneResponse* response,
    MyClosure done) {
  // for (auto& pair : name2worker_) {
  //  WorkerInitDataPlaneRequest init_dp_req;
  //  WorkerInitDataPlaneResponse init_dp_resp;
  //  ::tensorflow::Status s =
  //      pair.second->WorkerInitDataPlane(&init_dp_req, &init_dp_resp);
  //  CHECK(s.ok());
  //}
  for (auto& pair : name2worker_) {
    struct Call {
      WorkerInitDataPlaneRequest init_dp_req;
      WorkerInitDataPlaneResponse init_dp_resp;
    };
    Call* call = new Call;

    auto cb = [call](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker Init Dataplane RPC succeeds";
      } else {
        LOG(INFO) << "Worker Init Dataplane RPC fails";
      }
      delete call;
    };
    pair.second->WorkerInitDataPlaneAsync(&call->init_dp_req,
                                          &call->init_dp_resp, cb);
  }
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
