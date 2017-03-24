#include "net/grpc_worker_cache.h"
#include "net/grpc_channel.h"
#include "net/worker_interface.h"

namespace oneflow {

class GrpcWorkerCache : public WorkerCacheInterface {
 public:
  GrpcWorkerCache(GrpcChannelCache* channel_cache) : channel_cache_(channel_cache){}
  ~GrpcWorkerCache(){}

 private:
  GrpcChannelCache* channel_cache_;
};

WorkerCacheInterface* NewGrpcWorkerCache(GrpcChannelCache* cc){
  return new GrpcWorkerCache(cc);
}

}
