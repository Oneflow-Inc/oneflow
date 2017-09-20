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

  void Barrier(const std::string& barrier_name);
  void Barrier(const std::string& barrier_name, int32_t barrier_num);

  TryLockResult TryLock(const std::string& name);
  void NotifyDone(const std::string& name);
  void WaitUntilDone(const std::string& name);

 private:
  CtrlCommNet() = default;
  CtrlService::Stub* GetMasterStub() { return stubs_[0].get(); }
  CtrlService::Stub* GetResponsibleStub(const std::string& key);

  std::unique_ptr<CtrlServer> ctrl_server_;
  std::vector<std::unique_ptr<CtrlService::Stub>> stubs_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() CtrlCommNet::Singleton()->Barrier(FILE_LINE_STR)

#define OF_ONCE_GUARD(name, ...)                                      \
  do {                                                                \
    TryLockResult lock_ret = CtrlCommNet::Singleton()->TryLock(name); \
    if (lock_ret == TryLockResult::kLocked) {                         \
      __VA_ARGS__;                                                    \
      CtrlCommNet::Singleton()->NotifyDone(name);                     \
    } else if (lock_ret == TryLockResult::kDone) {                    \
    } else if (lock_ret == TryLockResult::kDoing) {                   \
      CtrlCommNet::Singleton()->WaitUntilDone(name);                  \
    } else {                                                          \
      UNEXPECTED_RUN();                                               \
    }                                                                 \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_CTRL_COMM_NETWORK_H_
