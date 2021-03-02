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

#ifndef ONEFLOW_CORE_RPC_INCLUDE_CLIENT_
#define ONEFLOW_CORE_RPC_INCLUDE_CLIENT_

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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_CLIENT_
