#ifndef ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
#define ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_

#include <grpc++/alarm.h>
#include <grpc++/server_builder.h>
#include "oneflow/core/control/ctrl_call.h"
#include "oneflow/core/common/function_traits.h"

namespace oneflow {

namespace {
template<size_t... Idx>
static std::tuple<std::function<void(CtrlCall<(CtrlMethod)Idx>*)>...> GetHandlerTuple(
    std::index_sequence<Idx...>) {
  return {};
}

}  // namespace

class CtrlServer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlServer);
  ~CtrlServer();

  CtrlServer();
  const std::string& this_machine_addr() { return this_machine_addr_; }

 private:
  void HandleRpcs();
  void Init();

  void EnqueueRequests() {
    for_each_i(handlers_, helper{this}, std::make_index_sequence<kCtrlMethodNum>{});
  }

  template<CtrlMethod kMethod>
  void EnqueueRequest() {
    constexpr const size_t I = (size_t)kMethod;
    auto handler = std::get<I>(handlers_);
    auto call = new CtrlCall<(CtrlMethod)I>();
    call->set_request_handler(std::bind(handler, call));
    grpc_service_->RequestAsyncUnary(I, call->mut_server_ctx(), call->mut_request(),
                                     call->mut_responder(), cq_.get(), cq_.get(), call);
  }

  template<typename F>
  void Add(F f) {
    using args_type = typename function_traits<F>::args_type;
    using arg_type =
        typename std::remove_pointer<typename std::tuple_element<0, args_type>::type>::type;

    std::get<arg_type::value>(handlers_) = std::move(f);
  }

  struct helper {
    helper(CtrlServer* s) : s_(s) {}
    template<typename T, typename V>
    void operator()(const T& t, V) {
      s_->EnqueueRequest<(CtrlMethod)V::value>();
    }

    CtrlServer* s_;
  };

  using HandlerTuple = decltype(GetHandlerTuple(std::make_index_sequence<kCtrlMethodNum>{}));

  HandlerTuple handlers_;
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
  HashMap<std::string, std::list<CtrlCall<CtrlMethod::kPullKV>*>> pending_kv_calls_;
  // IncreaseCount, EraseCount
  HashMap<std::string, int32_t> count_;

  bool is_first_connect_;
  std::string this_machine_addr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_SERVER_H_
