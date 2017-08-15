#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_

#include "oneflow/core/distributed_runtime/master.pb.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Master {
 public:
  Master(const ServerDef& server_def,
         ::grpc::CompletionQueue* completion_queue);
  ~Master();

  // Convenient typedef for a closure passing a Status
  typedef std::function<void(const ::tensorflow::Status&)> MyClosure;

  ::tensorflow::Status SendJob(SendJobRequest* request,
                               SendJobResponse* response, MyClosure done);

  ::tensorflow::Status MasterInitDataPlane(
      MasterInitDataPlaneRequest* request,
      MasterInitDataPlaneResponse* response, MyClosure done);

 private:
  void ParseServerDef();
  void CreateWorkerCache();

  // The overall server configuration.
  const ServerDef server_def_;
  std::string this_node_name_;
  std::unordered_map<std::string, ClusterNode> name2node_def_;
  std::unordered_map<std::string, std::shared_ptr<GrpcRemoteWorker>>
      name2worker_;

  ::grpc::CompletionQueue* cq_;
};  // Master
}  // namespace oneflow
#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
