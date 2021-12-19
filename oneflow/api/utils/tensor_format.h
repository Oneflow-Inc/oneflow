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

// Some code of this file is copied from PyTorch aten/src/ATen/core/Formatting.h

#ifndef ONEFLOW_API_UTILS_TENSOR_FORMAT_H_
#define ONEFLOW_API_UTILS_TENSOR_FORMAT_H_

#include <cstdint>
#include <ostream>
#include <vector>

namespace oneflow {

// saves/restores number formatting inside scope
struct FormatGuard {                                                                 // NOLINT
  FormatGuard(std::ostream& out) : out(out), saved(nullptr) { saved.copyfmt(out); }  // NOLINT
  ~FormatGuard() { out.copyfmt(saved); }

 private:
  std::ostream& out;
  std::ios saved;
};

std::ostream& FormatTensorData(std::ostream& stream, const double* data,
                               const std::vector<int64_t>& dim_vec, int64_t linesize);

}  // namespace oneflow

#endif // ONEFLOW_API_UTILS_TENSOR_FORMAT_H_
