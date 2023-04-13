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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_PRIMITIVE_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_PRIMITIVE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

namespace ep {
namespace primitive {

class Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Primitive);
  Primitive() = default;
  virtual ~Primitive() = default;
};

template<typename PrimitiveT>
class Factory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Factory);
  Factory() = default;
  virtual ~Factory() = default;

  using PrimitiveType = PrimitiveT;
};

template<typename FactoryType, typename... Args>
static std::unique_ptr<typename FactoryType::PrimitiveType> NewPrimitive(DeviceType device_type,
                                                                         Args&&... args) {
  if (!IsClassRegistered<DeviceType, FactoryType>(device_type)) { return nullptr; }
  std::unique_ptr<FactoryType> factory = NewObjUniquePtr<DeviceType, FactoryType>(device_type);
  if (!factory) { return nullptr; }
  return factory->New(std::forward<Args>(args)...);
}

#define REGISTER_PRIMITIVE_FACTORY(device, Base, Derived) \
  REGISTER_CLASS(DeviceType, device, Base, Derived)

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_PRIMITIVE_H_
