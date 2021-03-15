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
#ifndef ONEFLOW_CORE_FRAMEWORK_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_OBJECT_H_

#include <functional>
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/python_interpreter_util.h"

namespace oneflow {

namespace compatible_py {

class Object {
 public:
  Object(int64_t object_id, const std::shared_ptr<ParallelDesc>& parallel_desc_symbol);
  virtual ~Object() = default;

  int64_t object_id() const;
  std::shared_ptr<ParallelDesc> parallel_desc_symbol() const;

 private:
  int64_t object_id_;
  std::shared_ptr<ParallelDesc> parallel_desc_symbol_;
};

class BlobObject : public Object {
 public:
  BlobObject(int64_t object_id, const std::shared_ptr<OpArgParallelAttribute>& op_arg_parallel_attr,
             const std::shared_ptr<OpArgBlobAttribute>& op_arg_blob_attr);
  ~BlobObject() override {
    if (!(CHECK_JUST(IsShuttingDown()))) { ForceReleaseAll(); }
  }

  std::shared_ptr<OpArgParallelAttribute> op_arg_parallel_attr() const;

  std::shared_ptr<OpArgBlobAttribute> op_arg_blob_attr() const;

  void add_releaser(const std::function<void(Object*)>& release);

  void ForceReleaseAll();

 private:
  std::shared_ptr<OpArgParallelAttribute> op_arg_parallel_attr_;
  std::shared_ptr<OpArgBlobAttribute> op_arg_blob_attr_;
  std::vector<std::function<void(Object*)>> release_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OBJECT_H_
