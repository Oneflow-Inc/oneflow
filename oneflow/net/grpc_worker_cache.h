#ifndef GRPC_WORKER_CACHE_H_
#define GRPC_WORKER_CACHE_H_

#include "net/worker_cache.h"
#include "net/grpc_channel.h"

namespace oneflow {
 
WorkerCacheInterface* NewGrpcWorkerCache(GrpcChannelCache* cc);

}

#endif
