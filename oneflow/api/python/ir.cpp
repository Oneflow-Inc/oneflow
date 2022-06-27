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

#ifdef WITH_MLIR

#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
#include <glog/logging.h>
#include "oneflow/api/python/of_api_registry.h"
#include "mlir/"

class LR_JIT {
  private:
    std::unordered_map<std::function_id, std::function> function_id2lr_func_;
}

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("ir", m) {
  m.def("load_jit_shared_lib",
        [](const std::string& lib_path) { MutSharedLibPaths()->insert(lib_path); });
  m.def("compile_and_register_lr_jit",
        [](const python_ast& ast, std::string function_id) { 
          ir = lower_ast_to_llvm(ast, function_id);
          jit = JitEngine::create(ir);
          std::function lr_func = (double base_lr, int64_t step)[] {
            jit.invoke(base_lr, step);
          }
          Global<LR_JIT>::get()->register(function_id, lr_func);
          Global<LR_JIT>::get()->invoke(function_id, 0.01, 19); // test it
         });
}

}  // namespace oneflow

#endif  // WITH_MLIR
