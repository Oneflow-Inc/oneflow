#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/platform.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNet);
  virtual ~CommNet() = default;

  static CommNet* Singleton() { return comm_network_ptr_; }

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual const void* RegisterMemory(void* dptr, size_t byte_size) = 0;
  virtual void UnRegisterMemory(const void* token) = 0;
  virtual void RegisterMemoryDone() = 0;

  virtual void EstablishNetwork() = 0;

  // Stream
  virtual void* NewActorReadId() = 0;
  virtual void DeleteActorReadId(void* actor_read_id) = 0;
  virtual void* Read(void* actor_read_id, int64_t src_machine_id,
                     const void* src_token, const void* dst_token) = 0;
  virtual void AddReadCallBack(void* actor_read_id, void* read_id,
                               std::function<void()> callback) = 0;
  virtual void AddReadCallBackDone(void* actor_read_id, void* read_id) = 0;
  virtual void ReadDone(void* read_done_id) = 0;

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;

 protected:
  CommNet() = default;
  static void set_comm_network_ptr(CommNet* val) { comm_network_ptr_ = val; }

 private:
  static CommNet* comm_network_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
