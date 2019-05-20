#ifndef ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_COMM_NETWORK_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

class CommNet {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNet);
  CommNet() = delete;
  virtual ~CommNet();

  // "RegisterMemory" will return a Token, after "RegisterMemoryDone",
  // we can use this token to use the "Read"
  virtual void* RegisterMemory(void* ptr, size_t byte_size) = 0;
  virtual void UnRegisterMemory(void* token) = 0;
  virtual void RegisterMemoryDone() = 0;

  // Stream
  void Read(int64_t stream_id, int64_t src_machine_id, void* src_token, void* dst_token);
  void AddReadCallBack(int64_t stream_id, std::function<void()> callback);
  void ReadDone(int64_t stream_id);

  //
  virtual void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) = 0;

 protected:
  CommNet(const Plan& plan);

  virtual void DoRead(int64_t stream_id, int64_t src_machine_id, void* src_token,
                      void* dst_token) = 0;
  const HashSet<int64_t>& peer_machine_id() { return peer_machine_id_; }

  Channel<std::function<void()>> ready_cbs_;

 private:
  friend class Global<CommNet>;
  struct CommNetItem {
    std::function<void()> callback;
    bool is_read;
    CommNetItem(const std::function<void()>& callback, bool is_read)
        : callback(callback), is_read(is_read) {}
  };
  void IssueCallBack(const std::function<void()>& cb);
  void AddWorkToStream(int64_t stream_id, const std::function<void()>& cb, bool is_read);
  HashSet<int64_t> peer_machine_id_;
  HashMap<int64_t, std::unique_ptr<std::mutex>> stream_id2stream_mtx_ptr_;
  HashMap<int64_t, std::queue<CommNetItem>> stream_id2stream_;
  std::thread ready_cb_poller_;
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
