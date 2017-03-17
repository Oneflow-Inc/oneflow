#ifndef GRPC_WORKER_CACHE_H_
#define GRPC_WORKER_CACHE_H_

#include "worker_cache.h"
#include "grpc_channel.h"

namespace oneflow {
 
WorkerCacheInterface* NewGrpcWorkerCache(GrpcChannelCache* cc);

}

#endif
