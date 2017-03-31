#ifndef MASTER_SESSION_INTERFACE_H_
#define MASTER_SESSION_INTERFACE_H_

namespace oneflow {

class MasterSessionInterface {
 public:
  MasterSessionInterface(){}
  virtual void Create() = 0;

}; 

}

#endif
