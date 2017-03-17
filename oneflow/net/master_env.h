#ifndef MASTER_ENV_H_
#define MASTER_ENV_H_

#include <functional>

namespace oneflow {

class WorkerCacheInterface;
class MasterSessionInterface;

struct MasterEnv {
  WorkerCacheInterface* worker_cache = nullptr;
  std::function<MasterSessionInterface*(MasterEnv*)> master_session_factory;
};

}

#endif
