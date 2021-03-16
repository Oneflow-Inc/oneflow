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

class CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCallIf);
  virtual ~CtrlCallIf() = default;

  virtual void Process() = 0;
  virtual void SendResponse() = 0;

 protected:
  CtrlCallIf() = default;

 private:
};

class RpcClient {
 public:
  RpcClient() {}
  virtual ~RpcClient() {}

  virtual void Barrier(const std::string& barrier_name) {}
  virtual void Barrier(const std::string& barrier_name, int32_t barrier_num) {}

  virtual TryLockResult TryLock(const std::string& name) = 0;
  virtual void NotifyDone(const std::string& name) {}
  virtual void WaitUntilDone(const std::string& name) {}

  virtual void PushKV(const std::string& k, std::function<void(std::string*)> VSetter) {}
  virtual void PushKV(const std::string& k, const std::string& v) {}
  virtual void PushKV(const std::string& k, const PbMessage& msg) {}
  virtual void PushMasterKV(const std::string& k, const PbMessage& msg) {}
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PushKVT(const std::string& k, T v) {
    PushKV(k, std::to_string(v));
  }

  virtual void ClearKV(const std::string& k) {}
  virtual void ClearMasterKV(const std::string& k) {}

  virtual void PullKV(const std::string& k, std::function<void(const std::string&)> VGetter) {}
  virtual void PullKV(const std::string& k, std::string* v) {}
  virtual void PullKV(const std::string& k, PbMessage* msg) {}
  virtual void PullMasterKV(const std::string& k, PbMessage* msg) {}
  template<typename T>
  typename std::enable_if<std::is_arithmetic<T>::value>::type PullKVT(const std::string& k, T* v) {
    std::string v_str;
    PullKV(k, &v_str);
    *v = oneflow_cast<T>(v_str);
  }

  virtual void PushActEvent(const ActEvent&) {}
  virtual void Clear() {}
};

class RpcServer {
 public:
  RpcServer() {}
  virtual ~RpcServer() {}
};

class CtrlClient : public RpcClient {
 public:
  CtrlClient() {}
  virtual ~CtrlClient() {}
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

template<CtrlMethod ctrl_method>
class CtrlCall final : public CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCall);
  CtrlCall() : status_(Status::kBeforeHandleRequest) {}
  ~CtrlCall() = default;

  static constexpr const size_t value = (size_t)ctrl_method;

  const CtrlRequest<ctrl_method>& request() const { return request_; }
  CtrlRequest<ctrl_method>* mut_request() { return &request_; }
  CtrlResponse<ctrl_method>* mut_response() { return &response_; }
  void set_request_handler(std::function<void()> val) { request_handler_ = val; }

  void Process() override {
    switch (status_) {
      case Status::kBeforeHandleRequest: {
        request_handler_();
        return;
      }
      case Status::kBeforeDelete: {
        delete this;
        return;
      }
    }
  }

  void SendResponse() override { status_ = Status::kBeforeDelete; }

 private:
  enum class Status { kBeforeHandleRequest, kBeforeDelete };

  Status status_;
  CtrlRequest<ctrl_method> request_;
  CtrlResponse<ctrl_method> response_;
  std::function<void()> request_handler_;
};

class CtrlServer {
 public:
  CtrlServer() {}
  virtual ~CtrlServer() {}
};

class RpcManager {
 public:
  RpcManager() {}
  virtual ~RpcManager() {}
  virtual void Bootstrap() {}
  virtual void CreateServer() {}
  virtual void CreateClient() {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_INCLUDE_BASE_CTRL_
