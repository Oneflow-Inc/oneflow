#ifndef GRPC_CHANNEL_H_
#define GRPC_CHANNEL_H_
#include <memory>
#include <set>

#include "grpc++/grpc++.h"

namespace oneflow{

class GrpcChannelSpec {
 public:
  struct HostPortsJob {
    std::string job_id;
    std::vector<std::string> host_ports;
    int tasks_per_replica;
  };
  void AddHostPortsJob(const std::string& job_id,
 		       std::vector<std::string>& host_ports,
		       int tasks_per_replica);
  const std::vector<HostPortsJob>& host_ports_jobs() const {
    return host_ports_jobs_;
  }
 private:
  std::vector<HostPortsJob> host_ports_jobs_;
  std::set<std::string> job_ids_;
};

class GrpcChannelCache {
 public:
  virtual ~GrpcChannelCache(){}
};

typedef std::shared_ptr<::grpc::Channel> SharedGrpcChannelPtr;
typedef std::function<SharedGrpcChannelPtr(std::string)> ChannelCreationFunction;

SharedGrpcChannelPtr NewHostPortGrpcChannel(const std::string& target);

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec, ChannelCreationFunction channel_function);
GrpcChannelCache* NewHostPortsGrpcChannelCache(
    const std::string& id, const std::vector<std::string>& host_ports,
    int tasks_per_replica, ChannelCreationFunction channel_function);

}
#endif
