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
#ifndef ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_
#define ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_

#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/ctrl_service.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

class CtrlClient final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlClient);
  ~CtrlClient();

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

  void PushActEvent(const ActEvent&);
  void Clear();

  int32_t IncreaseCount(const std::string& k, int32_t v);
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  void EraseCount(const std::string& k);

 private:
  friend class Global<CtrlClient>;
  CtrlClient();
  void LoadServer(const std::string& server_addr, CtrlService::Stub* stub);
  void PushMasterKV(const std::string& k, std::function<void(std::string*)> VSetter);
  void PullMasterKV(const std::string& k, std::function<void(const std::string&)> VGetter);
  CtrlService::Stub* GetMasterStub() { return stubs_[0].get(); }
  CtrlService::Stub* GetThisStub();
  CtrlService::Stub* GetResponsibleStub(const std::string& key);

  std::vector<std::unique_ptr<CtrlService::Stub>> stubs_;
  std::mutex done_names_mtx_;
  HashSet<std::string> done_names_;

  bool need_heartbeat_thread_stop_;
  std::mutex need_heartbeat_thread_stop_mtx_;
  std::thread heartbeat_thread_;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)

#define OF_ENV_BARRIER() Global<CtrlClient>::Get()->Barrier(FILE_LINE_STR)
#define OF_SESSION_BARRIER()                        \
  Global<CtrlClient>::Get()->Barrier(FILE_LINE_STR, \
                                     Global<ResourceDesc, ForSession>::Get()->TotalMachineNum())

static void OfCallOnce(const std::string& name, std::function<void()> f) {
  TryLockResult lock_ret = Global<CtrlClient>::Get()->TryLock(name);
  if (lock_ret == TryLockResult::kLocked) {
    f();
    Global<CtrlClient>::Get()->NotifyDone(name);
  } else if (lock_ret == TryLockResult::kDone) {
  } else if (lock_ret == TryLockResult::kDoing) {
    Global<CtrlClient>::Get()->WaitUntilDone(name);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename Self, typename F, typename Arg, typename... Args>
static void OfCallOnce(const std::string& name, Self self, F f, Arg&& arg, Args&&... args) {
  std::function<void()> fn =
      std::bind(f, self, std::forward<Arg>(arg), std::forward<Args>(args)...);
  OfCallOnce(name, std::move(fn));
}

template<typename Self, typename F>
static void OfCallOnce(const std::string& name, Self self, F f) {
  std::function<void()> fn = std::bind(f, self, name);
  OfCallOnce(name, std::move(fn));
}

template<typename F, typename Arg, typename... Args>
static void OfCallOnce(const std::string& name, F f, Arg&& arg, Args&&... args) {
  std::function<void()> fn = std::bind(f, std::forward<Arg>(arg), std::forward<Args>(args)...);
  OfCallOnce(name, std::move(fn));
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CLIENT_H_
