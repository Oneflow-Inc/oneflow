#include "oneflow/core/distributed_runtime/worker.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common_runtime/runtime.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"

namespace oneflow {

Worker::Worker(
    const std::string& this_node_name, Network* data_net,
    const std::unordered_map<std::string, std::shared_ptr<GrpcRemoteWorker>>&
        name2worker)
    : this_node_name_(this_node_name),
      data_net_(data_net),
      name2worker_(name2worker) {}

Worker::~Worker() {}

::tensorflow::Status Worker::SendPlan(SendPlanRequest* request,
                                      SendPlanResponse* response,
                                      MyClosure done) {
  std::string str_plan;
  PrintProtoToString(request->plan(), &str_plan);
  LOG(INFO) << str_plan;

  plan_ = request->plan();
  // Plan plan = request->plan();
  // oneflow::runtime::Runtime::Singleton()->Run(plan, this_node_name_);

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerConnectDataPlane(
    WorkerConnectDataPlaneRequest* request,
    WorkerConnectDataPlaneResponse* response, MyClosure done) {
  data_net_->ConnectTopology();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitRuntime(
    WorkerInitRuntimeRequest* request, WorkerInitRuntimeResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerInitRuntime";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitModel(WorkerInitModelRequest* request,
                                             WorkerInitModelResponse* response,
                                             MyClosure done) {
  LOG(INFO) << "WorkerInitModel";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerActivateActor(
    WorkerActivateActorRequest* request, WorkerActivateActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerActivateActor";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegstToInc(
    WorkerSendRemoteRegstToIncRequest* request,
    WorkerSendRemoteRegstToIncResponse* response, MyClosure done) {
  LOG(INFO) << "WorkerSendRemoteRegstToInc";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegstToDec(
    WorkerSendRemoteRegstToDecRequest* request,
    WorkerSendRemoteRegstToDecResponse* response, MyClosure done) {
  LOG(INFO) << "WorkerSendRemoteRegstToDec";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerStartActor(
    WorkerStartActorRequest* request, WorkerStartActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerStartActor";
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitDataPlane(
    WorkerInitDataPlaneRequest* request, WorkerInitDataPlaneResponse* response,
    MyClosure done) {
  // data_net_->ConnectTopology();
  LOG(INFO) << "Worker Init DataPlane done";

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

}  // namespace oneflow
