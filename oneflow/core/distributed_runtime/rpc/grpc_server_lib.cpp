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

#include "oneflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <cstring>
#include <limits>
#include <memory>
#include <thread>

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "grpc++/server_builder.h"
#include "grpc/support/alloc.h"

#include "oneflow/core/distributed_runtime/local_master.h"
#include "oneflow/core/distributed_runtime/master.h"
#include "oneflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_master_service.h"
#include "oneflow/core/distributed_runtime/rpc/grpc_worker_service.h"
#include "oneflow/core/distributed_runtime/server_lib.h"
#include "oneflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mem.h"

namespace oneflow {

namespace {

// Define an option subclass in order to disable SO_REUSEPORT for the
// server socket.
class NoReusePortOption : public ::grpc::ServerBuilderOption {
 public:
  void UpdateArguments(::grpc::ChannelArguments* args) override {
    args->SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
  }

  void UpdatePlugins(std::vector<std::unique_ptr<::grpc::ServerBuilderPlugin>>*
                         plugins) override {}
};
}  // namespace

GrpcServer::GrpcServer(const ServerDef& server_def)
    : server_def_(server_def), state_(NEW) {}

GrpcServer::~GrpcServer() {
  // TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  delete master_service_;
  delete worker_service_;
}

::tensorflow::Status GrpcServer::Init() {
  ::tensorflow::mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);

  ParseServerDef();
  CreateWorkerCache();

  data_net_ = GetRdmaInstance();
  data_net_->InitOnly(my_machine_id_, net_topo_);

  std::string server_address =
      ::tensorflow::strings::StrCat(bound_ip_, ":", bound_port_);
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  builder.SetMaxMessageSize(std::numeric_limits<int32_t>::max());

  master_impl_ = CreateMaster();
  master_service_ = NewGrpcMasterService(master_impl_.get(), &builder);

  worker_impl_ = CreateWorker();
  worker_service_ = NewGrpcWorkerService(worker_impl_.get(), &builder);

  server_ = builder.BuildAndStart();

  if (!server_) {
    return ::tensorflow::errors::Unknown("Could not start gRPC server");
  }

  return ::tensorflow::Status::OK();
}

void GrpcServer::ParseServerDef() {
  this_node_name_ = server_def_.this_node_name();

  int32_t node_num = server_def_.cluster_def().cluster_node_size();
  for (int32_t i = 0; i < node_num; ++i) {
    std::string node_name =
        server_def_.cluster_def().cluster_node(i).node_name();
    ClusterNode cluster_node = server_def_.cluster_def().cluster_node(i);
    CHECK(
        name2node_def_.insert(std::make_pair(node_name, cluster_node)).second);

    int32_t ctrl_port_n;
    std::string ctrl_port_str = cluster_node.ctrl_plane_addr().port();
    if (!::tensorflow::strings::safe_strto32(ctrl_port_str, &ctrl_port_n)) {
      LOG(FATAL) << "Could not parse port for local server from "
                 << ctrl_port_str;
    }

    int32_t data_port_n;
    std::string data_port_str = cluster_node.data_plane_addr().port();
    if (!::tensorflow::strings::safe_strto32(data_port_str, &data_port_n)) {
      LOG(FATAL) << "Could not parse port for local server from "
                 << data_port_str;
    }

    if (node_name == this_node_name_) {
      my_machine_id_ = i;
      bound_ip_ = cluster_node.ctrl_plane_addr().addr();
      bound_port_ = ctrl_port_n;
    }
    NetworkTopology::Node node;
    node.machine_id = i;
    node.address = cluster_node.data_plane_addr().addr();
    node.port = data_port_n;
    net_topo_.all_nodes.push_back(node);
    name2id_.insert({node_name, i});
  }

  for (auto& node : net_topo_.all_nodes) {
    for (auto& neighbor : net_topo_.all_nodes) {
      if (neighbor.machine_id != node.machine_id) {
        node.neighbors.insert(neighbor.machine_id);
      }
    }
  }
}

void GrpcServer::CreateWorkerCache() {
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
        new GrpcRemoteWorker(worker_channel, &completion_queue_));
    CHECK(name2worker_.insert(std::make_pair(name, remote_worker)).second);
    CHECK(name2id_.count(name));
    int64_t machine_id = name2id_[name];
    CHECK(id2worker_.insert(std::make_pair(machine_id, remote_worker)).second);
  }
}

::tensorflow::Status GrpcServer::Start() {
  ::tensorflow::mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      master_thread_ =
          std::thread(&AsyncServiceInterface::HandleRPCsLoop, master_service_);
      master_do_thread_ =
          std::thread(&AsyncServiceInterface::DoWorkLoop, master_service_);
      worker_thread_ =
          std::thread(&AsyncServiceInterface::HandleRPCsLoop, worker_service_);
      worker_do_thread_ =
          std::thread(&AsyncServiceInterface::DoWorkLoop, worker_service_);

      async_request_done_thread_ = std::thread([this]() {
        void* tag;
        bool ok;
        while (completion_queue_.Next(&tag, &ok)) {
          GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
          callback_tag->OnCompleted(ok);
        }
      });

      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      return ::tensorflow::Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return ::tensorflow::Status::OK();
    case STOPPED:
      return ::tensorflow::errors::FailedPrecondition("Server has stopped.");
    default: CHECK(false);
  }
  return ::tensorflow::Status();
}

::tensorflow::Status GrpcServer::Stop() {
  ::tensorflow::mutex_lock l(mu_);
  switch (state_) {
    case NEW: state_ = STOPPED; return ::tensorflow::Status::OK();
    case STARTED:
      // return ::tensorflow::errors::Unimplemented(
      //    "Clean shutdown is not currently implemented");
      master_service_->Shutdown();
      worker_service_->Shutdown();
      completion_queue_.Shutdown();
      return ::tensorflow::Status::OK();
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return ::tensorflow::Status::OK();
    default: CHECK(false);
  }
  return ::tensorflow::Status();
}

::tensorflow::Status GrpcServer::Join() {
  ::tensorflow::mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return ::tensorflow::Status::OK();
    case STARTED:
    case STOPPED:
      master_thread_.join();
      master_do_thread_.join();
      worker_thread_.join();
      worker_do_thread_.join();
      async_request_done_thread_.join();
      return ::tensorflow::Status::OK();
    default: CHECK(false);
  }
  return ::tensorflow::Status();
}

const std::string GrpcServer::target() const {
  return ::tensorflow::strings::StrCat("grpc://localhost:", bound_port_);
}

std::unique_ptr<Master> GrpcServer::CreateMaster() {
  return std::unique_ptr<Master>(new Master(id2worker_));
}

std::unique_ptr<Worker> GrpcServer::CreateWorker() {
  return std::unique_ptr<Worker>(
      new Worker(my_machine_id_, this_node_name_, data_net_, id2worker_));
}

/* static */
::tensorflow::Status GrpcServer::Create(
    const ServerDef& server_def, std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<GrpcServer> ret(new GrpcServer(server_def));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return ::tensorflow::Status::OK();
}

namespace {

class GrpcServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ::tensorflow::ServerDef& server_def) override {
    return server_def.protocol() == "grpc";
  }

  ::tensorflow::Status NewServer(
      const ::tensorflow::ServerDef& server_def,
      std::unique_ptr<ServerInterface>* out_server) override {
    // return GrpcServer::Create(server_def, Env::Default(), out_server);
    return ::tensorflow::Status();
  }
};

// Registers a `ServerFactory` for `GrpcServer` instances.
class GrpcServerRegistrar {
 public:
  GrpcServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    memset(&alloc_fns, 0, sizeof(alloc_fns));
    alloc_fns.malloc_fn = ::tensorflow::port::Malloc;
    alloc_fns.realloc_fn = ::tensorflow::port::Realloc;
    alloc_fns.free_fn = ::tensorflow::port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("GRPC_SERVER", new GrpcServerFactory());
  }
};
static GrpcServerRegistrar registrar;

}  // namespace
}  // namespace oneflow
