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

  ::oneflow::runtime::Runtime::Singleton()->SetPlan(request->plan());
  ::oneflow::runtime::Runtime::Singleton()->SetThisMachineName(this_node_name_);

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

  ::oneflow::runtime::Runtime::Singleton()->InitRuntime();

  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerInitModel(WorkerInitModelRequest* request,
                                             WorkerInitModelResponse* response,
                                             MyClosure done) {
  LOG(INFO) << "WorkerInitModel";
  ::oneflow::runtime::Runtime::Singleton()->InitModel();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerActivateActor(
    WorkerActivateActorRequest* request, WorkerActivateActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerActivateActor";
  ::oneflow::runtime::Runtime::Singleton()->ActivateActor();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegst(
    WorkerSendRemoteRegstRequest* request,
    WorkerSendRemoteRegstResponse* response, MyClosure done) {
  LOG(INFO) << "WorkerSendRemoteRegst";
  ::oneflow::runtime::Runtime::Singleton()->SendRemoteRegstToInc();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerSendRemoteRegstToConsumer(
    WorkerSendRemoteRegstToConsumerRequest* request,
    WorkerSendRemoteRegstToConsumerResponse* response, MyClosure done) {
  LOG(INFO) << "WorkerSendRemoteRegstToDec";
  ::oneflow::runtime::Runtime::Singleton()->SendRemoteRegstToDec();
  done(::tensorflow::Status());
  return ::tensorflow::Status::OK();
}

::tensorflow::Status Worker::WorkerStartActor(
    WorkerStartActorRequest* request, WorkerStartActorResponse* response,
    MyClosure done) {
  LOG(INFO) << "WorkerStartActor";
  ::oneflow::runtime::Runtime::Singleton()->StartActor();
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
