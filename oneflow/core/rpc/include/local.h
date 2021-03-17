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
#ifndef ONEFLOW_CORE_RPC_INCLUDE_LOCAL_H_
#define ONEFLOW_CORE_RPC_INCLUDE_LOCAL_H_

#include <string>
#include <unordered_map>
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/rpc/include/base.h"

namespace oneflow {

class LocalCtrlClient : public CtrlClient {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalCtrlClient);
  LocalCtrlClient(const ProcessCtx& process_ctx);
  ~LocalCtrlClient() = default;

  void Barrier(const std::string& barrier_name) override;
  void Barrier(const std::string& barrier_name, int32_t barrier_num) override;

  TryLockResult TryLock(const std::string& name) override;
  void NotifyDone(const std::string& name) override;
  void WaitUntilDone(const std::string& name) override;

  void PushKV(const std::string& k, std::function<void(std::string*)> VSetter) override;
  void PushKV(const std::string& k, const std::string& v) override;
  void PushKV(const std::string& k, const PbMessage& msg) override;
  void PushMasterKV(const std::string& k, const PbMessage& msg) override;

  void ClearKV(const std::string& k) override;
  void ClearMasterKV(const std::string& k) override;

  void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) override;
  void PullKV(const std::string& k, std::string* v) override;
  void PullKV(const std::string& k, PbMessage* msg) override;
  void PullMasterKV(const std::string& k, PbMessage* msg) override;
  void PushActEvent(const ActEvent&) override {}
  void Clear() override;

  HashSet<std::string> done_names_;
  std::mutex done_names_mtx_;
  std::condition_variable done_names_cv_;
  HashMap<std::string, std::string> kv_;
  std::mutex kv_mtx_;
  std::condition_variable kv_cv_;
};

class LocalRpcManager : public RpcManager {
 public:
  LocalRpcManager() {}
  ~LocalRpcManager();
  Maybe<void> Bootstrap();
  Maybe<void> CreateServer();
  Maybe<void> CreateClient();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_LOCAL_H_
