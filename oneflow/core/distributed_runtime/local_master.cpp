/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "oneflow/core/distributed_runtime/local_master.h"

#include <unordered_map>

// #include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/platform/mutex.h"

namespace oneflow {

// namespace {
// Status WaitForNotification(CallOptions* call_options,
//                           const int64 default_timeout_in_ms, Notification* n)
//                           {
//  int64 timeout_in_ms = call_options->GetTimeout();
//  if (timeout_in_ms == 0) {
//    timeout_in_ms = default_timeout_in_ms;
//  }
//  if (timeout_in_ms > 0) {
//    int64 timeout_in_us = timeout_in_ms * 1000;
//    bool notified = WaitForNotificationWithTimeout(n, timeout_in_us);
//    if (!notified) {
//      call_options->StartCancel();
//      // The call has borrowed pointers to the request and response
//      // messages, so we must still wait for the call to complete.
//      n->WaitForNotification();
//      return errors::DeadlineExceeded("Operation timed out.");
//    }
//  } else {
//    n->WaitForNotification();
//  }
//  return Status::OK();
//}
//}  // namespace

// LocalMaster::LocalMaster(Master* master_impl, const int64
// default_timeout_in_ms)
//    : master_impl_(master_impl),
//      default_timeout_in_ms_(default_timeout_in_ms) {}

::tensorflow::Status LocalMaster::SendJob(const SendJobRequest* request,
                                          SendJobResponse* response) {
  ::tensorflow::Status ret;
  // master_impl_->CreateSession(request, response, [&n, &ret](const Status& s)
  // {
  //  ret.Update(s);
  //  n.Notify();
  //});
  // TF_RETURN_IF_ERROR(
  //    WaitForNotification(call_options, default_timeout_in_ms_, &n));
  return ret;
}

}  // namespace oneflow
