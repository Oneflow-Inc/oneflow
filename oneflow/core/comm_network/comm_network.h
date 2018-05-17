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
  CommNet() = delete;
  virtual ~CommNet() = default;

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual void* RegisterMemory(void* ptr, size_t byte_size) = 0;
  virtual void UnRegisterMemory(void* token) = 0;
  virtual void RegisterMemoryDone() = 0;

  // Stream
  void* NewActorReadId();
  void DeleteActorReadId(void* actor_read_id);
  void* Read(void* actor_read_id, int64_t src_machine_id, void* src_token, void* dst_token);
  void AddReadCallBack(void* actor_read_id, void* read_id, std::function<void()> callback);
  void AddReadCallBackDone(void* read_id);
  void ReadDone(void* read_id);

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;

 protected:
  CommNet(const Plan& plan);

  virtual void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) = 0;
  const HashSet<int64_t>& peer_machine_id() { return peer_machine_id_; }

 private:
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
  friend class Global<CommNet>;
  int8_t IncreaseDoneCnt(ReadContext*);
  void FinishOneRead(ReadContext*);

  HashSet<int64_t> peer_machine_id_;
};

template<typename MemDescType>
class CommNetIf : public CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetIf);
  CommNetIf() = delete;
  CommNetIf(const Plan& plan) : CommNet(plan) {}
  virtual ~CommNetIf() {}

  void* RegisterMemory(void* ptr, size_t byte_size) override {
    MemDescType* mem_desc = NewMemDesc(ptr, byte_size);
    std::unique_lock<std::mutex> lck(mem_descs_mtx_);
    CHECK(mem_descs_.insert(mem_desc).second);
    return mem_desc;
  }

  void UnRegisterMemory(void* token) override {
    MemDescType* mem_desc = static_cast<MemDescType*>(token);
    delete mem_desc;
    std::unique_lock<std::mutex> lck(mem_descs_mtx_);
    CHECK_EQ(mem_descs_.erase(mem_desc), 1);
  }

 protected:
  virtual MemDescType* NewMemDesc(void* ptr, size_t byte_size) = 0;
  const HashSet<MemDescType*>& mem_descs() { return mem_descs_; }

 private:
  std::mutex mem_descs_mtx_;
  HashSet<MemDescType*> mem_descs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
