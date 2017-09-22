#ifndef ONEFLOW_CORE_COMM_NETWORK_DATA_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_DATA_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/platform.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class DataCommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataCommNet);
  virtual ~DataCommNet() = default;

  static DataCommNet* Singleton() { return data_comm_network_ptr_; }

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual const void* RegisterMemory(void* dptr, size_t byte_size) = 0;
  virtual void UnRegisterMemory(const void* token) = 0;
  virtual void RegisterMemoryDone() = 0;

  // Stream
  virtual void* CreateStream() = 0;
  virtual void Read(void* stream_id, const void* src_token,
                    const void* dst_token) = 0;
  virtual void AddCallBack(void* stream_id, std::function<void()>) = 0;

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;

 protected:
  DataCommNet() = default;
  static void set_comm_network_ptr(DataCommNet* val) {
    data_comm_network_ptr_ = val;
  }

 private:
  static DataCommNet* data_comm_network_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_DATA_COMM_NETWORK_H_
