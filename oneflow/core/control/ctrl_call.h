#ifndef ONEFLOW_CORE_CONTROL_CTRL_CALL_H_
#define ONEFLOW_CORE_CONTROL_CTRL_CALL_H_

#include "oneflow/core/control/ctrl_service.h"

namespace oneflow {

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

template<typename RequestMessageType, typename ResponseMessageType>
class CtrlCall final : public CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCall);
  CtrlCall()
      : status_(Status::kBeforeHandleRequest), responder_(&server_ctx_) {}
  ~CtrlCall() = default;

  const RequestMessageType& request() const { return request_; }

  RequestMessageType* mut_request() { return &request_; }
  ResponseMessageType* mut_response() { return &response_; }
  grpc::ServerContext* mut_server_ctx() { return &server_ctx_; }
  grpc::ServerAsyncResponseWriter<ResponseMessageType>* mut_responder() {
    return &responder_;
  }
  void set_request_handler(std::function<void()> val) {
    request_handler_ = val;
  }

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

  void SendResponse() override {
    responder_.Finish(response_, grpc::Status::OK, this);
    status_ = Status::kBeforeDelete;
  }

 private:
  enum class Status { kBeforeHandleRequest, kBeforeDelete };

  Status status_;
  RequestMessageType request_;
  ResponseMessageType response_;
  grpc::ServerContext server_ctx_;
  grpc::ServerAsyncResponseWriter<ResponseMessageType> responder_;
  std::function<void()> request_handler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CALL_H_
