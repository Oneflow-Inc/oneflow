#ifndef ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_
#define ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_

#include <string>
#include <map>

#include "grpc++/grpc++.h"

namespace oneflow {

class GrpcChannelCache {
 public:
  GrpcChannelCache();
  ~GrpcChannelCache(); 

  void CreateChannelCache();

  std::map<std::string, std::shared_ptr<::grpc::Channel>> channel_map_;
};


}


#endif  // ONEFLOW_CORE_DISTRIBUTED_RUNTIME_GRPC_CHANNEL_CACHE_H_
