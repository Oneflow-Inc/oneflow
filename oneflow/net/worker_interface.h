#ifndef WORKER_INTERFACE_H_
#define WORKER_INTERFACE_H_

namespace oneflow {

class WorkerInterface {
 public:
  virtual void role_register() = 0;
};
}

#endif
