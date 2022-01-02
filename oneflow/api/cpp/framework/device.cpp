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

#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"

namespace oneflow_api {

namespace of = oneflow;

Device::Device(const std::string& type_or_type_with_device_id)
    : device_(std::make_shared<of::Symbol<of::Device>>(
        of::Device::ParseAndNew(type_or_type_with_device_id).GetOrThrow())) {}

Device::Device(const std::string& type, int64_t device_id)
    : device_(
        std::make_shared<of::Symbol<of::Device>>(of::Device::New(type, device_id).GetOrThrow())) {}

const std::string& Device::type() const { return (*device_)->type(); }

int64_t Device::device_id() const { return (*device_)->device_id(); }

bool Device::operator==(const Device& rhs) const { return *device_ == *rhs.device_; }
bool Device::operator!=(const Device& rhs) const { return *device_ != *rhs.device_; }

}  // namespace oneflow_api
