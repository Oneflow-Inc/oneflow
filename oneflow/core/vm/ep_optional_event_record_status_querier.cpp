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
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"

namespace oneflow {
namespace vm {

void EpOptionalEventRecordStatusQuerier::SetLaunched(ep::Stream* stream) {
  CHECK(!launched_);
  if (ep_event_) {
    ep_event_->mut_device()->SetAsActiveDevice();
    stream->RecordEvent(ep_event_->mut_event());
  }
  launched_ = true;
}

EpOptionalEventRecordStatusQuerier::~EpOptionalEventRecordStatusQuerier() {}

}  // namespace vm
}  // namespace oneflow
