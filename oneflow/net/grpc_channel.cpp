#include "net/grpc_channel.h"

#include "grpc++/create_channel.h"

namespace oneflow {

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
