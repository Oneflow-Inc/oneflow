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

GrpcChannelCache* NewGrpcChannelCache(ChannelCreationFunction channel_function);
GrpcChannelCache* NewHostPortsGrpcChannelCache(ChannelCreationFunction channel_function);

}
#endif
