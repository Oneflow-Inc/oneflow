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
#ifndef ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
#define ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_

#include <memory>
#include "oneflow/core/common/sized_buffer_view.h"

namespace oneflow {

struct ZeroOnlyZipUtil final {
  void ZipToSizedBuffer(const char* data, size_t size, SizedBufferView* sized_buffer);
  void UnzipToExpectedSize(const SizedBufferView& size_buffer, char* data, size_t expected_size);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ZERO_ONLY_ZIP_H_
