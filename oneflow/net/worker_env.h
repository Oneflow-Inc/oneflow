#ifndef WORKER_ENV_H_
#define WORKER_ENV_H_

namespace oneflow{

class WorkerCacheInterface;

struct WorkerEnv{
 WorkerCacheInterface* worker_cache = nullptr;
};

}

#endif
