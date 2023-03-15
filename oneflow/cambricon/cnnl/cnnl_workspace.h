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
#ifndef ONEFLOW_CAMBRICON_CNNL_CNNL_WORKSPACE_H_
#define ONEFLOW_CAMBRICON_CNNL_CNNL_WORKSPACE_H_

#include "oneflow/cambricon/ep/mlu_stream.h"

namespace oneflow {

class CnnlWorkspace {
 public:
  CnnlWorkspace(ep::MluStream* stream, size_t workspace_size = 0);
  ~CnnlWorkspace();

  void resize(size_t workspace_size);

  size_t size() const { return size_; }

  void* dptr() { return workspace_dptr_; }
  const void* dptr() const { return workspace_dptr_; }

 private:
  ep::MluStream* mlu_stream_;
  size_t size_;
  size_t capacity_;
  char* workspace_dptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_CNNL_CNNL_WORKSPACE_H_
