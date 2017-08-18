/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
#include <memory>
#include <thread>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"

#include "oneflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "oneflow/core/distributed_runtime/server_def.pb.h"
#include "oneflow/core/distributed_runtime/server_lib.h"
#include "oneflow/core/network/network.h"
#include "oneflow/core/network/network_topology.h"

#include "tensorflow/core/platform/mutex.h"

namespace oneflow {

class Master;
class Worker;

class GrpcServer : public ServerInterface {
 protected:
  GrpcServer(const ServerDef& server_def);

 public:
  static ::tensorflow::Status Create(
      const ServerDef& server_def,
      std::unique_ptr<ServerInterface>* out_server);

  // Destruction is only supported in the factory method. Clean
  // shutdown is not currently implemented for this server type.
  virtual ~GrpcServer();

  // Implementations of ServerInterface methods.
  ::tensorflow::Status Start() override;
  ::tensorflow::Status Stop() override;
  ::tensorflow::Status Join() override;
  const std::string target() const override;

 protected:
  ::tensorflow::Status Init();
  void ParseServerDef();
  virtual std::unique_ptr<Master> CreateMaster();
  virtual std::unique_ptr<Worker> CreateWorker();

  void CreateWorkerCache();

  // Returns the port to which this server is bound.
  // This method may only be called after `this->Init()` returns successfully.
  int bound_port() const { return bound_port_; }
  std::string bound_ip() const { return bound_ip_; }
  const ServerDef& server_def() const { return server_def_; }

 private:
  // The overall server configuration.
  const ServerDef server_def_;

  std::string this_node_name_;
  // The IP address which this server uses
  std::string bound_ip_;
  // The port to which this server is bound.
  int bound_port_ = 0;

  // For data plane network
  NetworkTopology net_topo_;
  int64_t my_machine_id_;
  Network* data_net_;

  // Guards state transitions.
  ::tensorflow::mutex mu_;

  // Represents the current state of the server, which changes as follows:
  //
  //                 Join()            Join()
  //                  ___               ___
  //      Start()     \ /    Stop()     \ /
  // NEW ---------> STARTED --------> STOPPED
  //   \                          /
  //    \________________________/
  //            Stop(), Join()
  enum State { NEW, STARTED, STOPPED };
  State state_;

  // Implementation of a TensorFlow master, and RPC polling thread.
  // MasterEnv master_env_;
  std::unique_ptr<Master> master_impl_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::thread master_thread_;
  std::thread master_do_thread_;

  //// Implementation of a TensorFlow worker, and RPC polling thread.
  std::unique_ptr<Worker> worker_impl_;
  AsyncServiceInterface* worker_service_ = nullptr;
  std::thread worker_thread_;
  std::thread worker_do_thread_;

  // CQ used for async request from master to workers
  ::grpc::CompletionQueue completion_queue_;
  std::thread async_request_done_thread_;

  std::unordered_map<std::string, ClusterNode> name2node_def_;
  std::unordered_map<std::string, std::shared_ptr<GrpcRemoteWorker>>
      name2worker_;
  std::unordered_map<int64_t, std::shared_ptr<GrpcRemoteWorker>> id2worker_;
  std::unordered_map<std::string, int64_t> name2id_;

  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
