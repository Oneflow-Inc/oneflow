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
#include <mutex>
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/object_storage.h"

namespace oneflow {

std::mutex global_parallel_sym2shared_opkernel_obj_mutex;

namespace {

HashMap<ParallelDesc, std::shared_ptr<compatible_py::Object>>*
GlobalParallelConfSym2SharedOpkernelObject() {
  static HashMap<ParallelDesc, std::shared_ptr<compatible_py::Object>>
      parallel_conf_symbol2shared_opkernel_object;
  return &parallel_conf_symbol2shared_opkernel_object;
}

}  // namespace

Maybe<bool> HasSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym) {
  std::unique_lock<std::mutex> lock(global_parallel_sym2shared_opkernel_obj_mutex);
  auto* parallel_conf_symbol2shared_opkernel_object = GlobalParallelConfSym2SharedOpkernelObject();
  return (*parallel_conf_symbol2shared_opkernel_object).find(*parallel_conf_sym)
         != (*parallel_conf_symbol2shared_opkernel_object).end();
}

Maybe<compatible_py::Object> GetOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym) {
  std::unique_lock<std::mutex> lock(global_parallel_sym2shared_opkernel_obj_mutex);
  auto* parallel_conf_symbol2shared_opkernel_object = GlobalParallelConfSym2SharedOpkernelObject();
  return (*parallel_conf_symbol2shared_opkernel_object).at(*parallel_conf_sym);
}

Maybe<void> SetSharedOpKernelObject4ParallelConfSymbol(
    const std::shared_ptr<ParallelDesc>& parallel_conf_sym,
    const std::shared_ptr<compatible_py::Object>& shared_opkernel_object) {
  std::unique_lock<std::mutex> lock(global_parallel_sym2shared_opkernel_obj_mutex);
  auto* parallel_conf_symbol2shared_opkernel_object = GlobalParallelConfSym2SharedOpkernelObject();
  CHECK_OR_RETURN((*parallel_conf_symbol2shared_opkernel_object).find(*parallel_conf_sym)
                  == (*parallel_conf_symbol2shared_opkernel_object).end());
  (*parallel_conf_symbol2shared_opkernel_object)[*parallel_conf_sym] = shared_opkernel_object;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
