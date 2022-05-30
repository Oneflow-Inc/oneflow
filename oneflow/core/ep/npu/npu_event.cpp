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
#include "oneflow/core/ep/npu/npu_event.h"
#include <iostream>
#ifdef WITH_NPU

namespace oneflow {

namespace ep {

NpuEvent::NpuEvent(unsigned int flags) : npu_event_{} {
  std::cout<<"NpuEvent::NpuEvent(unsigned int flags)"<<std::endl;
  OF_NPU_CHECK(aclrtCreateEventWithFlag(&npu_event_, flags));
}

NpuEvent::~NpuEvent() { OF_NPU_CHECK(aclrtDestroyEvent(npu_event_)); }

Maybe<bool> NpuEvent::QueryDone() {
  std::cout<<"NpuEvent::QueryDone()"<<std::endl;
  aclrtEventStatus status;
  aclError err = aclrtQueryEvent(npu_event_,&status);
  if (err == ACL_SUCCESS) {
    return Maybe<bool>(true);
  }
  else {
    return Error::RuntimeError() << err;
  }
}

Maybe<void> NpuEvent::Sync() {
  std::cout<<"NpuEvent::Sync()"<<std::endl;
  aclError err = aclrtSynchronizeEvent(npu_event_);
  if (err == ACL_SUCCESS) {
    return Maybe<void>::Ok();
  } else {
    return Error::RuntimeError() << err;
  }
}

aclrtEvent NpuEvent::npu_event() { return npu_event_; }

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_NPU
