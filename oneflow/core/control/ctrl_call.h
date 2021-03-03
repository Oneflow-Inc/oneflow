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

template<CtrlMethod ctrl_method>
class CtrlCall final : public CtrlCallIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CtrlCall);
  CtrlCall() : status_(Status::kBeforeHandleRequest), responder_(&server_ctx_) {}
  ~CtrlCall() = default;

  static constexpr const size_t value = (size_t)ctrl_method;

  const CtrlRequest<ctrl_method>& request() const { return request_; }
  CtrlRequest<ctrl_method>* mut_request() { return &request_; }
  CtrlResponse<ctrl_method>* mut_response() { return &response_; }
  grpc::ServerContext* mut_server_ctx() { return &server_ctx_; }
  const grpc::ServerContext& server_ctx() const { return server_ctx_; }
  grpc::ServerAsyncResponseWriter<CtrlResponse<ctrl_method>>* mut_responder() {
    return &responder_;
  }
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

  void SendResponse() override {
    responder_.Finish(response_, grpc::Status::OK, this);
    status_ = Status::kBeforeDelete;
  }

 private:
  enum class Status { kBeforeHandleRequest, kBeforeDelete };

  Status status_;
  CtrlRequest<ctrl_method> request_;
  CtrlResponse<ctrl_method> response_;
  grpc::ServerContext server_ctx_;
  grpc::ServerAsyncResponseWriter<CtrlResponse<ctrl_method>> responder_;
  std::function<void()> request_handler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CONTROL_CTRL_CALL_H_
