#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_

#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Master {
 public:
  Master(const std::unordered_map<int64_t, std::shared_ptr<GrpcRemoteWorker>>&
             id2worker);
  ~Master();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendJob(SendJobRequest* request,
                               SendJobResponse* response, MyClosure done);

  ::tensorflow::Status MasterConnectDataPlane(
      MasterConnectDataPlaneRequest* request,
      MasterConnectDataPlaneResponse* response, MyClosure done);

  ::tensorflow::Status MasterInitRuntime(MasterInitRuntimeRequest* request,
                                         MasterInitRuntimeResponse* response,
                                         MyClosure done);

  ::tensorflow::Status MasterInitModel(MasterInitModelRequest* request,
                                       MasterInitModelResponse* response,
                                       MyClosure done);

  ::tensorflow::Status MasterActivateActor(
      MasterActivateActorRequest* request,
      MasterActivateActorResponse* response, MyClosure done);

  ::tensorflow::Status MasterSendRemoteRegst(
      MasterSendRemoteRegstRequest* request,
      MasterSendRemoteRegstResponse* response, MyClosure done);

  ::tensorflow::Status MasterStartActor(MasterStartActorRequest* request,
                                        MasterStartActorResponse* response,
                                        MyClosure done);

  ::tensorflow::Status MasterInitDataPlane(
      MasterInitDataPlaneRequest* request,
      MasterInitDataPlaneResponse* response, MyClosure done);

 private:
  const std::unordered_map<int64_t, std::shared_ptr<GrpcRemoteWorker>>&
      id2worker_;
};  // Master
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
