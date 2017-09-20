#ifndef ONEFLOW_CORE_COMM_NETWORK_CTRL_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_CTRL_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/comm_network/ctrl_server.h"

namespace oneflow {

class CtrlCommNet final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCommNet);
  ~CtrlCommNet() = default;

  OF_SINGLETON(CtrlCommNet);

  void Init();

  void Barrier(const std::string& barrier_name) {}

  // 0 : locked
  // 1 : done
  // 2 : doing
  int32_t TryLock(const std::string& name) {
    // TODO
    static HashSet<std::string> locked_name;
    if (locked_name.find(name) != locked_name.end()) {
      return 1;
    } else {
      locked_name.insert(name);
      return 0;
    }
  }
  void NotifyDone(const std::string& name) {}
  void WaitForLockDone(const std::string& name) {}

 private:
  CtrlCommNet() = default;
  CtrlService::Stub* GetMasterStub() { return stubs_[0].get(); }

  std::unique_ptr<CtrlServer> ctrl_server_;
  std::vector<std::unique_ptr<CtrlService::Stub>> stubs_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() CtrlCommNet::Singleton()->Barrier(FILE_LINE_STR)

#define OF_ONCE_GUARD(name, ...)                                \
  do {                                                          \
    int32_t lock_ret = CtrlCommNet::Singleton()->TryLock(name); \
    if (lock_ret == 0) {                                        \
      __VA_ARGS__;                                              \
      CtrlCommNet::Singleton()->NotifyDone(name);               \
    } else if (lock_ret == 1) {                                 \
    } else if (lock_ret == 2) {                                 \
      CtrlCommNet::Singleton()->WaitForLockDone(name);          \
    } else {                                                    \
      UNEXPECTED_RUN();                                         \
    }                                                           \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_CTRL_COMM_NETWORK_H_
