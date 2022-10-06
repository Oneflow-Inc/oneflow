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
#ifndef ONEFLOW_CORE_EP_PRIMITIVE_ADD_H_
#define ONEFLOW_CORE_EP_PRIMITIVE_ADD_H_

#include "oneflow/core/ep/include/primitive/primitive.h"

namespace oneflow {

namespace ep {
namespace primitive {

class Add : public Primitive {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Add);
  Add() = default;
  ~Add() override = default;

  virtual void Launch(Stream* stream, const void* const* srcs, size_t arity, void* dst,
                      size_t count) = 0;
  virtual void Launch(Stream* stream, const void* src0, const void* src1, void* dst, size_t count);
};

class AddFactory : public Factory<Add> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddFactory);
  AddFactory() = default;
  ~AddFactory() override = default;

  virtual std::unique_ptr<Add> New(DataType data_type) = 0;
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_PRIMITIVE_ADD_H_
