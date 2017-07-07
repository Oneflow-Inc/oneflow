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
  virtual void Barrier(const std::string& barrier_name) = 0;

 protected:
  CommNetwork() = default;
  static void set_comm_network_ptr(CommNetwork* val) {
    comm_network_ptr_ = val;
  }

 private:
  static CommNetwork* comm_network_ptr_;
};

#define OF_MACRO_TRICK1(x) #x
#define OF_MACRO_TRICK2(x) OF_MACRO_TRICK1(x)
#define OF_BARRIER() \
  CommNetwork::Singleton()->Barrier(__FILE__ ":" OF_MACRO_TRICK2(__LINE__))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
