#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_

#include <array>
#include <grpc++/alarm.h>
#include <grpc++/server_builder.h>
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/control/ctrl_call.h"

namespace oneflow {

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  CtrlServer() = delete;
  ~CtrlServer();

  CtrlServer(const std::string& server_addr);

 private:
  void HandleRpcs();

  template<typename... Args>
  void init(Args... args) {
    static_assert(sizeof...(Args) == kCtrlMethodNum, "must equal");
    arr_ = {reinterpret_cast<Member>(args)...};
  }

  using RequestType =
      std::tuple<LoadServerRequest, BarrierRequest, TryLockRequest, NotifyDoneRequest,
                 WaitUntilDoneRequest, PushKVRequest, ClearKVRequest, PullKVRequest,
                 PushActEventRequest, ClearRequest, IncreaseCountRequest, EraseCountRequest,
                 PushAvgActIntervalRequest>;
  using ResponseType =
      std::tuple<LoadServerResponse, BarrierResponse, TryLockResponse, NotifyDoneResponse,
                 WaitUntilDoneResponse, PushKVResponse, ClearKVResponse, PullKVResponse,
                 PushActEventResponse, ClearResponse, IncreaseCountResponse, EraseCountResponse,
                 PushAvgActIntervalResponse>;
  typedef void (CtrlServer::*Member)(void*);
  std::array<Member, kCtrlMethodNum> arr_;

  template<std::size_t I = 0, typename T>
  typename std::enable_if<I == array_size<T>::size>::type EnqueueRequests(T& arr){};

  template<std::size_t I = 0, typename T>
      typename std::enable_if < I<array_size<T>::size>::type EnqueueRequests(T& arr) {
    EnqueueRequest<I>();
    EnqueueRequests<I + 1>(arr);
  }

  template<size_t I>
  void EnqueueRequest();

  template<CtrlMethod I>
  void EnqueueRequest() {
    EnqueueRequest<(size_t)I>();
  }

  void LoadServerHandler(CtrlCall<LoadServerRequest, LoadServerResponse>* call);
  void BarrierHandler(CtrlCall<BarrierRequest, BarrierResponse>* call);
  void TryLockHandler(CtrlCall<TryLockRequest, TryLockResponse>* call);
  void NotifyDoneHandler(CtrlCall<NotifyDoneRequest, NotifyDoneResponse>* call);
  void WaitUntilDoneHandler(CtrlCall<WaitUntilDoneRequest, WaitUntilDoneResponse>* call);
  void PushKVHandler(CtrlCall<PushKVRequest, PushKVResponse>* call);
  void ClearKVHandler(CtrlCall<ClearKVRequest, ClearKVResponse>* call);
  void PullKVHandler(CtrlCall<PullKVRequest, PullKVResponse>* call);
  void PushActEventHandler(CtrlCall<PushActEventRequest, PushActEventResponse>* call);
  void ClearHandler(CtrlCall<ClearRequest, ClearResponse>* call);
  void IncreaseCountHandler(CtrlCall<IncreaseCountRequest, IncreaseCountResponse>* call);
  void EraseCountHandler(CtrlCall<EraseCountRequest, EraseCountResponse>* call);
  void PushAvgActIntervalHandler(
      CtrlCall<PushAvgActIntervalRequest, PushAvgActIntervalResponse>* call);

  std::unique_ptr<CtrlService::AsyncService> grpc_service_;
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::thread loop_thread_;
  // Barrier
  HashMap<std::string, std::pair<std::list<CtrlCallIf*>, int32_t>> barrier_calls_;
  // TryLock, NotifyDone, WaitUntilDone
  HashMap<std::string, void*> name2lock_status_;
  // PushKV, ClearKV, PullKV
  HashMap<std::string, std::string> kv_;
  HashMap<std::string, std::list<CtrlCall<PullKVRequest, PullKVResponse>*>> pending_kv_calls_;
  // IncreaseCount, EraseCount
  HashMap<std::string, int32_t> count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
