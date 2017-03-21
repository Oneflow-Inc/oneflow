#ifndef ONEFLOW_ASYNC_SERVICE_INTERFACE_H_
#define ONEFLOW_ASYNC_SERVICE_INTERFACE_H_

namespace oneflow{

class AsyncServiceInterface{
 public:
  virtual void HandleRPCsLoop() = 0; 
  virtual ~AsyncServiceInterface() {}
  
};
}
#endif
