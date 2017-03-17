#ifndef GRPC_CHANNEL_H_
#define GRPC_CHANNEL_H_
#include <memory>
#include <set>

#include "grpc++/grpc++.h"

namespace oneflow{

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
