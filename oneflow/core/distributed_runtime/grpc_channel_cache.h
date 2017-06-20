#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_

#include <string>
#include <map>

#include "grpc++/grpc++.h"

#include "oneflow/core/proto/cluster_config.pb.h"

using ::grpc::Channel;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ClientContext;

namespace oneflow {

class GrpcChannelCache {
 public:
  explicit GrpcChannelCache(oneflow::ClusterSpec cluster_spec);
  ~GrpcChannelCache();

  void CreateChannelCache();
  std::shared_ptr<::grpc::Channel> FindChannel(const std::string& address);

  std::map<std::string, std::shared_ptr<::grpc::Channel>> channel_map_;

  oneflow::ClusterSpec cluster_spec_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_
