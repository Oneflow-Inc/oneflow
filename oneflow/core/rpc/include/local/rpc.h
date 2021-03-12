/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_CORE_RPC_INCLUDE_LOCAL_RPC_
#define ONEFLOW_CORE_RPC_INCLUDE_LOCAL_RPC_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/control.pb.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/rpc/include/base/rpc.h"
#include "oneflow/core/common/function_traits.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class RpcClient : RpcClientBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RpcClient);
  virtual ~RpcClient() = default;

  void Barrier(const std::string& barrier_name);
  void Barrier(const std::string& barrier_name, int32_t barrier_num);

  TryLockResult TryLock(const std::string& name);
  void NotifyDone(const std::string& name);
  void WaitUntilDone(const std::string& name);

  void PushKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PushKV(const std::string& k, const std::string& v);
  void PushKV(const std::string& k, const PbMessage& msg);
  void PushMasterKV(const std::string& k, const PbMessage& msg);
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PushKVT(const std::string& k, T v) {
    PushKV(k, std::to_string(v));
  }

  void ClearKV(const std::string& k);
  void ClearMasterKV(const std::string& k);

  void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter);
  void PullKV(const std::string& k, std::string* v);
  void PullKV(const std::string& k, PbMessage* msg);
  void PullMasterKV(const std::string& k, PbMessage* msg);
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PullKVT(const std::string& k, T* v) {
    std::string v_str;
    PullKV(k, &v_str);
    *v = oneflow_cast<T>(v_str);
  }

  void PushActEvent(const ActEvent&){};
  void Clear();

  int32_t IncreaseCount(const std::string& k, int32_t v);
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  void EraseCount(const std::string& k);

 protected:
  RpcClient(){};
  void PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PullMasterKV(const std::string& k, std::function<void(const std::string&)> VGetter);

  HashMap<std::string, std::string> kv_;
  std::mutex done_names_mtx_;
  std::condition_variable done_names_cv_;
  HashSet<std::string> done_names_;
};

class RpcServer : RpcServerBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RpcServer);
  virtual ~RpcServer();

 protected:
  RpcServer() {}
  void HandleRpcs();
  void Init();

  void EnqueueRequests() {}

  std::thread loop_thread_;
  // TryLock, NotifyDone, WaitUntilDone
  HashMap<std::string, void*> name2lock_status_;
  // PushKV, ClearKV, PullKV
  HashMap<std::string, std::string> kv_;
  // IncreaseCount, EraseCount
  HashMap<std::string, int32_t> count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_LOCAL_RPC_
