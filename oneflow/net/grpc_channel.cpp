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

GrpcChannelCache* NewGrpcChannelCache(ChannelCreationFunction channel_func){
  std::vector<GrpcChannelCache*> caches;
  caches.push_back(NewHostPortsGrpcChannelCache(channel_func));
  return caches[0];
}

class HostPortsGrpcChannelCache : public GrpcChannelCache{
 public:
  HostPortsGrpcChannelCache(ChannelCreationFunction channel_func)
	: channel_func_(channel_func){}
  ~HostPortsGrpcChannelCache() override {}

 private:
  const ChannelCreationFunction channel_func_;
};

GrpcChannelCache* NewHostPortsGrpcChannelCache(ChannelCreationFunction channel_func){
  return new HostPortsGrpcChannelCache(channel_func);
}

}
