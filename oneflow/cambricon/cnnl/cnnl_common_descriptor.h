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
#ifndef ONEFLOW_CAMBRICON_CNNL_CNNL_COMMON_DESCRIPTOR_H_
#define ONEFLOW_CAMBRICON_CNNL_CNNL_COMMON_DESCRIPTOR_H_

#include <memory>
#include <vector>

#include "oneflow/cambricon/common/mlu_util.h"
#include "cnnl.h"

// Modified from Cambricon catch for PyTorch.
// https://github.com/Cambricon/catch/blob/main/torch_mlu/csrc/aten/cnnl/cnnlCommonDescriptors.h

namespace oneflow {

template<typename T, cnnlStatus_t (*dtor)(T*)>
struct CnnlDescriptorDeleter {
  void operator()(T* ptr) {
    if (ptr != nullptr) { OF_CNNL_CHECK(dtor(ptr)); }
  }
};

template<typename T, cnnlStatus_t (*ctor)(T**), cnnlStatus_t (*dtor)(T*)>
class CnnlDescriptor {
 public:
  CnnlDescriptor() = default;

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const { return desc_.get(); }
  T* desc() { return desc_.get(); }

  // Use CnnlDescriptor() to access the underlying desciptor pointer
  // if you intend to modify what it points to This will ensure
  // that the descriptor is initialized.
  // Code in this file will use this function.
  T* mut_desc() {
    init();
    return desc_.get();
  }

 protected:
  void init() {
    if (desc_ == nullptr) {
      T* ptr;
      OF_CNNL_CHECK(ctor(&ptr));
      desc_.reset(ptr);
    }
  }

 private:
  std::unique_ptr<T, CnnlDescriptorDeleter<T, dtor> > desc_;
};

void convertShapeAndStride(std::vector<int>& shape_info, std::vector<int>& stride_info);

}  // namespace oneflow

#endif  // ONEFLOW_CAMBRICON_CNNL_CNNL_COMMON_DESCRIPTOR_H_
