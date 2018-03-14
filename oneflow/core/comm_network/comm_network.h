#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/platform.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/machine_context.h"

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

  // Stream
  void* NewActorReadId();
  void DeleteActorReadId(void* actor_read_id);
  void* Read(void* actor_read_id, int64_t src_machine_id, const void* src_token,
             const void* dst_token);
  void AddReadCallBack(void* actor_read_id, void* read_id,
                       std::function<void()> callback);
  void AddReadCallBackDone(void* read_id);
  void ReadDone(void* read_id);

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;
  const HashSet<int64_t>& peer_machine_id() { return peer_machine_id_; }

 protected:
  CommNet() = default;
  static void set_comm_network_ptr(CommNet* val) { comm_network_ptr_ = val; }

  struct ActorReadContext;
  struct ReadContext {
    ActorReadContext* actor_read_ctx;
    std::list<std::function<void()>> cbl;
    std::mutex done_cnt_mtx;
    int8_t done_cnt;
  };
  struct ActorReadContext {
    std::mutex read_ctx_list_mtx;
    std::list<ReadContext*> read_ctx_list;
  };

  virtual void DoRead(void* read_id, int64_t src_machine_id,
                      const void* src_token, const void* dst_token) = 0;
  void GenConnectionInfo(const Plan& plan);

 private:
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneRead(ReadContext*);

  HashSet<int64_t> peer_machine_id_;
  static CommNet* comm_network_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
