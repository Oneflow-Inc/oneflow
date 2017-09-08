#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

class CommNetwork {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetwork);
  virtual ~CommNetwork() = default;

  static CommNetwork* Singleton() { return comm_network_ptr_; }

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
    return 0;  // TODO
  }
  void UnLock(const std::string& name) {}
  void WaitForLockDone(const std::string& name) {}

 protected:
  CommNetwork() = default;
  static void set_comm_network_ptr(CommNetwork* val) {
    comm_network_ptr_ = val;
  }

 private:
  static CommNetwork* comm_network_ptr_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_BARRIER() CommNetwork::Singleton()->Barrier(FILE_LINE_STR)

#define OF_ONCE_GUARD(name, ...)                                \
  do {                                                          \
    int32_t lock_ret = CommNetwork::Singleton()->TryLock(name); \
    if (lock_ret == 0) {                                        \
      __VA_ARGS__;                                              \
      CommNetwork::Singleton()->UnLock(name);                   \
    } else if (lock_ret == 1) {                                 \
    } else if (lock_ret == 2) {                                 \
      CommNetwork::Singleton()->WaitForLockDone(name);          \
    } else {                                                    \
      UNEXPECTED_RUN();                                         \
    }                                                           \
  } while (0)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
