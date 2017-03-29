#include <unordered_map>
#include <iostream>
#include "net/grpc_channel.h"

#include "grpc++/create_channel.h"
#include "re2/re2.h"

namespace oneflow {

namespace {
  RE2* kHostPortRE = new RE2("([^:/]+):(\\d+)");
}

void GrpcChannelSpec::AddHostPortsJob(const std::string& job_id,
                                      std::vector<std::string>& host_ports,
                                      int tasks_per_replica) {
  if(!job_ids_.insert(job_id).second) {
    return;
  }
  HostPortsJob job;
  job.job_id = job_id;
  for (auto& host_port : host_ports) {
    std::string host;
    int port;
    if (!RE2::FullMatch(host_port, *kHostPortRE, &host, &port)) {
      return;
    }
    job.host_ports = host_ports;
    job.tasks_per_replica = tasks_per_replica;
    host_ports_jobs_.push_back(job);
  }

}

SharedGrpcChannelPtr NewHostPortGrpcChannel(const std::string& target){
  ::grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_MESSAGE_LENGTH, std::numeric_limits<int>::max());
  return ::grpc::CreateCustomChannel(
      target, ::grpc::InsecureChannelCredentials(), args);
}

GrpcChannelCache* NewGrpcChannelCache(const GrpcChannelSpec& spec, ChannelCreationFunction channel_func){
  const int num_jobs = spec.host_ports_jobs().size();
  std::vector<GrpcChannelCache*> caches;
  caches.reserve(num_jobs);
  for(const GrpcChannelSpec::HostPortsJob& job: spec.host_ports_jobs()){
    caches.push_back(NewHostPortsGrpcChannelCache(
          job.job_id, job.host_ports, job.tasks_per_replica, channel_func));
  }
  return caches[0];
}

class CachingGrpcChannelCache : public GrpcChannelCache {
 public:
  CachingGrpcChannelCache() {}
  ~CachingGrpcChannelCache() {}
 private:
  std::unordered_map<std::string, SharedGrpcChannelPtr> channels_;
};

class HostPortsGrpcChannelCache : public GrpcChannelCache{
 public:
  HostPortsGrpcChannelCache(const std::string& job_id,
                            const std::vector<std::string>& host_ports,
                            int tasks_per_replica,
                            ChannelCreationFunction channel_func)
    : job_id_(job_id),
      host_ports_(),
      tasks_per_replica_(tasks_per_replica),
      channel_func_(channel_func) {}

  ~HostPortsGrpcChannelCache() override {}

 private:
  const std::string job_id_;
  std::vector<std::string> host_ports_;
  const int tasks_per_replica_;
  const ChannelCreationFunction channel_func_;
};

GrpcChannelCache* NewHostPortsGrpcChannelCache(
    const std::string& job_id, const std::vector<std::string>& host_ports,
    int tasks_per_replica, ChannelCreationFunction channel_func){
  return new HostPortsGrpcChannelCache(job_id, host_ports, tasks_per_replica, channel_func);
}

}
