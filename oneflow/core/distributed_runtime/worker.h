#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_

#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/network/network.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Worker {
 public:
  Worker(
      const std::string& this_node_name, Network* data_net,
      const std::unordered_map<std::string, std::shared_ptr<GrpcRemoteWorker>>&
          name2worker);
  ~Worker();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendPlan(SendPlanRequest* request,
                                SendPlanResponse* response, MyClosure done);

  ::tensorflow::Status WorkerConnectDataPlane(
      WorkerConnectDataPlaneRequest* request,
      WorkerConnectDataPlaneResponse* response, MyClosure done);

  ::tensorflow::Status WorkerInitRuntime(WorkerInitRuntimeRequest* request,
                                         WorkerInitRuntimeResponse* response,
                                         MyClosure done);

  ::tensorflow::Status WorkerInitModel(WorkerInitModelRequest* request,
                                       WorkerInitModelResponse* response,
                                       MyClosure done);

  ::tensorflow::Status WorkerActivateActor(
      WorkerActivateActorRequest* request,
      WorkerActivateActorResponse* response, MyClosure done);

  ::tensorflow::Status WorkerSendRemoteRegst(
      WorkerSendRemoteRegstRequest* request,
      WorkerSendRemoteRegstResponse* response, MyClosure done);

  ::tensorflow::Status WorkerStartActor(WorkerStartActorRequest* request,
                                        WorkerStartActorResponse* response,
                                        MyClosure done);

  ::tensorflow::Status WorkerInitDataPlane(
      WorkerInitDataPlaneRequest* request,
      WorkerInitDataPlaneResponse* response, MyClosure done);

 private:
  std::string this_node_name_;
  Plan plan_;
  Network* data_net_;

  const std::unordered_map<std::string, std::shared_ptr<GrpcRemoteWorker>>&
      name2worker_;
};  // Worker
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_H_
