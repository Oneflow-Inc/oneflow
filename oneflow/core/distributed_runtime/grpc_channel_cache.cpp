#include "oneflow/core/distributed_runtime/grpc_channel_cache.h"


namespace oneflow {

GrpcChannelCache::GrpcChannelCache(oneflow::ClusterSpec cluster_spec)
  : cluster_spec_(cluster_spec) {}
GrpcChannelCache::~GrpcChannelCache() {}

void GrpcChannelCache::CreateChannelCache() {
  std::string src_ip = "";
  std::string src_port = "";
  std::string src_ip_port = src_ip + ":" + src_port;
  for (auto& node_info : cluster_spec_.node_info()) {
    std::string dst_ip = node_info.ip();
    std::string dst_port = node_info.port();

    std::string dst_ip_port = dst_ip + ":" + dst_port;
    std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateChannel(dst_ip_port, ::grpc::InsecureChannelCredentials());
    channel_map_.insert({dst_ip_port, channel});
  }
}

std::shared_ptr<::grpc::Channel> GrpcChannelCache::FindChannel(
    const std::string& address) {
  if (channel_map_.find(address) != channel_map_.end()) {
    return channel_map_[address];
  }
}


}  // namespace oneflow
