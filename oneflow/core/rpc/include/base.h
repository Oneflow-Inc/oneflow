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
#ifndef ONEFLOW_CORE_RPC_INCLUDE_BASE_CTRL_
#define ONEFLOW_CORE_RPC_INCLUDE_BASE_CTRL_

#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/control.pb.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"

namespace oneflow {

#define CTRL_METHOD_SEQ               \
  OF_PP_MAKE_TUPLE_SEQ(LoadServer)    \
  OF_PP_MAKE_TUPLE_SEQ(Barrier)       \
  OF_PP_MAKE_TUPLE_SEQ(TryLock)       \
  OF_PP_MAKE_TUPLE_SEQ(NotifyDone)    \
  OF_PP_MAKE_TUPLE_SEQ(WaitUntilDone) \
  OF_PP_MAKE_TUPLE_SEQ(PushKV)        \
  OF_PP_MAKE_TUPLE_SEQ(ClearKV)       \
  OF_PP_MAKE_TUPLE_SEQ(PullKV)        \
  OF_PP_MAKE_TUPLE_SEQ(PushActEvent)  \
  OF_PP_MAKE_TUPLE_SEQ(Clear)         \
  OF_PP_MAKE_TUPLE_SEQ(IncreaseCount) \
  OF_PP_MAKE_TUPLE_SEQ(EraseCount)

#define CatRequest(method) method##Request,
#define CatReqponse(method) method##Response,
#define CatEnum(method) k##method,
#define CatName(method) "/oneflow.CtrlService/" OF_PP_STRINGIZE(method),

#define MAKE_META_DATA()                                                                       \
  enum class CtrlMethod { OF_PP_FOR_EACH_TUPLE(CatEnum, CTRL_METHOD_SEQ) };                    \
  static const char* g_method_name[] = {OF_PP_FOR_EACH_TUPLE(CatName, CTRL_METHOD_SEQ)};       \
  using CtrlRequestTuple = std::tuple<OF_PP_FOR_EACH_TUPLE(CatRequest, CTRL_METHOD_SEQ) void>; \
  using CtrlResponseTuple = std::tuple<OF_PP_FOR_EACH_TUPLE(CatReqponse, CTRL_METHOD_SEQ) void>;

MAKE_META_DATA()

constexpr const size_t kCtrlMethodNum = OF_PP_SEQ_SIZE(CTRL_METHOD_SEQ);

template<CtrlMethod ctrl_method>
using CtrlRequest =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlRequestTuple>::type;

template<CtrlMethod ctrl_method>
using CtrlResponse =
    typename std::tuple_element<static_cast<size_t>(ctrl_method), CtrlResponseTuple>::type;

inline const char* GetMethodName(CtrlMethod method) {
  return g_method_name[static_cast<int32_t>(method)];
}

class CtrlClient {
 public:
  explicit CtrlClient(const ProcessCtx& process_ctx);
  CtrlClient() = default;
  virtual ~CtrlClient() = default;

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

  virtual void PushActEvent(const ActEvent&) = 0;
  virtual void Clear() = 0;
  virtual int32_t IncreaseCount(const std::string& k, int32_t v) = 0;
  int32_t IncreaseCount(const std::string& k) { return IncreaseCount(k, 1); }
  virtual void EraseCount(const std::string& k) = 0;
};

#define FILE_LINE_STR __FILE__ ":" OF_PP_STRINGIZE(__LINE__)
#define OF_ENV_BARRIER() Global<CtrlClient>::Get()->Barrier(FILE_LINE_STR)
#define OF_SESSION_BARRIER()          \
  Global<CtrlClient>::Get()->Barrier( \
      FILE_LINE_STR, Global<ResourceDesc, ForSession>::Get()->process_ranks().size())

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

class RpcManager {
 public:
  RpcManager() = default;
  virtual ~RpcManager() = default;
  virtual Maybe<void> Bootstrap() = 0;
  virtual Maybe<void> CreateServer() = 0;
  virtual Maybe<void> CreateClient() = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_BASE_CTRL_
