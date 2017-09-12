#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

class CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNet);
  virtual ~CommNet() = default;

  static CommNet* Singleton() { return comm_network_ptr_; }

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual const void* RegisterMemory(void* dptr) = 0;
  virtual void RegisterMemoryDone() = 0;
  virtual void Read(const void* src_token, const void* dst_token,
                    std::function<void()> callback) = 0;

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;
  virtual void SetCallbackForReceivedActorMsg(
      std::function<void(const ActorMsg&)> callback) = 0;

  //
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

 protected:
  CommNet() = default;
  static void set_comm_network_ptr(CommNet* val) { comm_network_ptr_ = val; }

 private:
  static CommNet* comm_network_ptr_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() CommNet::Singleton()->Barrier(FILE_LINE_STR)

#define OF_ONCE_GUARD(name, ...)                            \
  do {                                                      \
    int32_t lock_ret = CommNet::Singleton()->TryLock(name); \
    if (lock_ret == 0) {                                    \
      __VA_ARGS__;                                          \
      CommNet::Singleton()->NotifyDone(name);               \
    } else if (lock_ret == 1) {                             \
    } else if (lock_ret == 2) {                             \
      CommNet::Singleton()->WaitForLockDone(name);          \
    } else {                                                \
      UNEXPECTED_RUN();                                     \
    }                                                       \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
