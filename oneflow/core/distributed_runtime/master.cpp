#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/compiler/compiler.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

#include "tensorflow/core/lib/core/blocking_counter.h"

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

  ::tensorflow::BlockingCounter blocking_counter(name2worker_.size());

  for (auto& pair : name2worker_) {
    struct Call {
      SendPlanRequest plan_req;
      SendPlanResponse plan_resp;
    };
    Call* call = new Call;
    *(call->plan_req.mutable_plan()) = response->plan();

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "SendPlan RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "SendPlan RPC fails: " << pair.first;
      }
      delete call;
    };
    pair.second->SendPlanAsync(&call->plan_req, &call->plan_resp, cb);
  }

  blocking_counter.Wait();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterConnectDataPlane(
    MasterConnectDataPlaneRequest* request,
    MasterConnectDataPlaneResponse* response, MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(name2worker_.size());

  for (auto& pair : name2worker_) {
    struct Call {
      WorkerConnectDataPlaneRequest connect_dp_req;
      WorkerConnectDataPlaneResponse connect_dp_resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker Connect Dataplane RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker Connect Dataplane RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerConnectDataPlaneAsync(&call->connect_dp_req,
                                             &call->connect_dp_resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterInitDataPlane(
    MasterInitDataPlaneRequest* request, MasterInitDataPlaneResponse* response,
    MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(name2worker_.size());

  for (auto& pair : name2worker_) {
    struct Call {
      WorkerInitDataPlaneRequest init_dp_req;
      WorkerInitDataPlaneResponse init_dp_resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker Init Dataplane RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker Init Dataplane RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerInitDataPlaneAsync(&call->init_dp_req,
                                          &call->init_dp_resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}
}  // namespace oneflow
