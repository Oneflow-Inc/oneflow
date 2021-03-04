#ifndef ONEFLOW_CORE_RPC_INCLUDE_BASE_
#define ONEFLOW_CORE_RPC_INCLUDE_BASE_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/control.pb.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

class RpcClientBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RpcClientBase);
  virtual ~RpcClientBase() = default;

  virtual void Barrier(const std::string& barrier_name) = 0;
  virtual void Barrier(const std::string& barrier_name, int32_t barrier_num) = 0;

  virtual TryLockResult TryLock(const std::string& name) = 0;
  virtual void NotifyDone(const std::string& name) = 0;
  virtual void WaitUntilDone(const std::string& name) = 0;

  virtual void PushKV(const std::string& k, std::function<void(std::string*)> VSetter) = 0;
  virtual void PushKV(const std::string& k, const std::string& v) = 0;
  virtual void PushKV(const std::string& k, const PbMessage& msg) = 0;
  virtual void PushMasterKV(const std::string& k, const PbMessage& msg) = 0;
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PushKVT(const std::string& k, T v) {
    PushKV(k, std::to_string(v));
  }

  virtual void ClearKV(const std::string& k) = 0;
  virtual void ClearMasterKV(const std::string& k) = 0;

  virtual void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) = 0;
  virtual void PullKV(const std::string& k, std::string* v) = 0;
  virtual void PullKV(const std::string& k, PbMessage* msg) = 0;
  virtual void PullMasterKV(const std::string& k, PbMessage* msg) = 0;
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PullKVT(const std::string& k, T* v) {
    std::string v_str;
    PullKV(k, &v_str);
    *v = oneflow_cast<T>(v_str);
  }

  virtual void PushActEvent(const ActEvent&) {}
  virtual void Clear();

  int32_t IncreaseCount(const std::string& k, int32_t v);
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  virtual void EraseCount(const std::string& k) = 0;

 protected:
  RpcClientBase() = default;
  virtual void PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter) = 0;
  virtual void PullMasterKV(const std::string& k,
                            std::function<void(const std::string&)> VGetter) = 0;
  std::mutex done_names_mtx_;
  HashSet<std::string> done_names_;
};

class RpcServerBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RpcServerBase);
  virtual ~RpcServerBase();

 protected:
  RpcServerBase() {}
  virtual void HandleRpcs() = 0;
  virtual void Init() = 0;

  virtual void EnqueueRequests() {}

  template<typename F>
  void Add(F f) {}

  std::thread loop_thread_;
  // Barrier
  // TryLock, NotifyDone, WaitUntilDone
  HashMap<std::string, void*> name2lock_status_;
  // PushKV, ClearKV, PullKV
  HashMap<std::string, std::string> kv_;
  // IncreaseCount, EraseCount
  HashMap<std::string, int32_t> count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_BASE_
