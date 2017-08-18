#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/compiler/compiler.h"
#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

#include "tensorflow/core/lib/core/blocking_counter.h"

namespace oneflow {

Master::Master(
    const std::unordered_map<int64_t, std::shared_ptr<GrpcRemoteWorker>>&
        id2worker)
    : id2worker_(id2worker) {}

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

  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
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
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
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

::tensorflow::Status Master::MasterInitRuntime(
    MasterInitRuntimeRequest* request, MasterInitRuntimeResponse* response,
    MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
    struct Call {
      WorkerInitRuntimeRequest req;
      WorkerInitRuntimeResponse resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker InitRuntime RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker InitRuntime RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerInitRuntimeAsync(&call->req, &call->resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterInitModel(MasterInitModelRequest* request,
                                             MasterInitModelResponse* response,
                                             MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
    struct Call {
      WorkerInitModelRequest req;
      WorkerInitModelResponse resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker InitModel RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker InitModel RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerInitModelAsync(&call->req, &call->resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterActivateActor(
    MasterActivateActorRequest* request, MasterActivateActorResponse* response,
    MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
    struct Call {
      WorkerActivateActorRequest req;
      WorkerActivateActorResponse resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker ActivateActor RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker ActivateActor RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerActivateActorAsync(&call->req, &call->resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterSendRemoteRegst(
    MasterSendRemoteRegstRequest* request,
    MasterSendRemoteRegstResponse* response, MyClosure done) {
  ::tensorflow::BlockingCounter inc_blocking_counter(id2worker_.size());
  for (auto& pair : id2worker_) {
    struct Call {
      WorkerSendRemoteRegstRequest req;
      WorkerSendRemoteRegstResponse resp;
    };
    Call* call = new Call;
    call->req.set_ascending_order(true);

    auto cb = [call, &inc_blocking_counter,
               &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker SendRemoteRegstToInc RPC succeeds";
        inc_blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker SendRemoteRegstToInc RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerSendRemoteRegstAsync(&call->req, &call->resp, cb);
  }
  inc_blocking_counter.Wait();

  ::tensorflow::BlockingCounter dec_blocking_counter(id2worker_.size());
  for (auto& pair : id2worker_) {
    struct Call {
      WorkerSendRemoteRegstRequest req;
      WorkerSendRemoteRegstResponse resp;
    };
    Call* call = new Call;
    call->req.set_ascending_order(false);

    auto cb = [call, &dec_blocking_counter,
               &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker SendRemoteRegstToDec RPC succeeds";
        dec_blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker SendRemoteRegstToDec RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerSendRemoteRegstAsync(&call->req, &call->resp, cb);
  }
  dec_blocking_counter.Wait();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterStartActor(
    MasterStartActorRequest* request, MasterStartActorResponse* response,
    MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
    struct Call {
      WorkerStartActorRequest req;
      WorkerStartActorResponse resp;
    };
    Call* call = new Call;

    auto cb = [call, &blocking_counter, &pair](const ::tensorflow::Status& s) {
      if (s.ok()) {
        LOG(INFO) << "Worker StartActor RPC succeeds";
        blocking_counter.DecrementCount();
      } else {
        LOG(FATAL) << "Worker StartActor RPC fails" << pair.first;
      }
      delete call;
    };
    pair.second->WorkerStartActorAsync(&call->req, &call->resp, cb);
  }

  blocking_counter.Wait();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Master::MasterInitDataPlane(
    MasterInitDataPlaneRequest* request, MasterInitDataPlaneResponse* response,
    MyClosure done) {
  ::tensorflow::BlockingCounter blocking_counter(id2worker_.size());

  for (auto& pair : id2worker_) {
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
