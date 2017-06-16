#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_

//#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/distributed_runtime/master_service.pb.h"
//#include "oneflow/core/distributed_runtime/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"

#include "tensorflow/core/lib/core/status.h"

namespace oneflow {

class Master {
 public:
  Master(GrpcChannelCache* channel_cache);
  ~Master();

  ::tensorflow::Status SendGraph(SendGraphRequest* request,
                 SendGraphResponse* response);
  void Barrier();

 private:
  //GrpcRemoteWorker* remote_worker_;
  GrpcChannelCache* channel_cache_;

};  // Master

}  // oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_H_
