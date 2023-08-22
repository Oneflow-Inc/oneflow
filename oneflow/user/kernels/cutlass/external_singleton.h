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

#ifdef WITH_CUTLASS

#include <vector>

#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"
#include "cutlass/library/operation_table.h"

#include "oneflow/core/common/util.h"

namespace cutlass {
namespace library {

/// Singleton instance stores a Manifest and Operation table
class ExternalSingleton {
 public:
  /// Manifest object
  std::vector<Manifest> manifest;

  /// Operation table referencing the Manifest
  OperationTable operation_table;

 public:
  ExternalSingleton();

  static ExternalSingleton& get();
};

#define ONEFLOW_CUTLASS_MANIFEST(m) ONEFLOW_CUTLASS_MANIFEST_IMPL(m, __COUNTER__)

#define ONEFLOW_CUTLASS_MANIFEST_IMPL(m, uuid)                        \
  static void OF_PP_CAT(_cutlass_manifest_, uuid)(Manifest & m);      \
  static int OF_PP_CAT(_cutlass_manifest_dummy_, uuid) = []() {       \
    auto& manifest = ExternalSingleton::get().manifest;               \
    manifest.resize(manifest.size() + 1);                             \
    OF_PP_CAT(_cutlass_manifest_, uuid)(manifest.back());             \
    ExternalSingleton::get().operation_table.append(manifest.back()); \
    return 0;                                                         \
  }();                                                                \
  void OF_PP_CAT(_cutlass_manifest_, uuid)(Manifest & m)

}  // namespace library
}  // namespace cutlass

#endif  // WITH_CUTLASS
