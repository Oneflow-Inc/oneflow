#ifndef MASTER_ENV_H_
#define MASTER_ENV_H_

namespace oneflow {

class WorkerCacheInterface;

struct MasterEnv {
  WorkerCacheInterface* worker_cache = nullptr;
};

}

#endif
