#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class CommNetwork final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetwork);
  ~CommNetwork() = default;

  OF_SINGLETON(CommNetwork);

  void Barrier(const std::string& barrier_name) {
    // TODO
  }

 private:
  CommNetwork() = default;
};

#define OF_MACRO_TRICK1(x) #x
#define OF_MACRO_TRICK2(x) OF_MACRO_TRICK1(x)
#define OF_BARRIER() \
  CommNetwork::Singleton()->Barrier(__FILE__ ":" OF_MACRO_TRICK2(__LINE__))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
