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
#include "oneflow/opencl/ep/cl_event.h"

#include "oneflow/opencl/common/cl_api.h"

namespace oneflow {
namespace ep {

clEvent::clEvent(unsigned int flags) : cl_event_{} {
  OF_CL_CHECK(clEventCreateWithFlags(&cl_event_, 0));
}

clEvent::~clEvent() { OF_CL_CHECK(clEventDestroy(cl_event_)); }

Maybe<bool> clEvent::QueryDone() {
  cl_int ret = clEventQuery(cl_event_);
  if (ret == CL_COMPLETE) {
    return Maybe<bool>(true);
  } else if (ret > 0) {
    return Maybe<bool>(false);
  } else {
    return Error::RuntimeError() << "clEvent::QueryDone() error";
  }
}

Maybe<void> clEvent::Sync() {
  cl_int err = clEventSynchronize(cl_event_);
  if (err == CL_SUCCESS) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << "clEvent::Sync() error";
  }
}

cl::Event* clEvent::cl_event() { return cl_event_; }

}  // namespace ep
}  // namespace oneflow
