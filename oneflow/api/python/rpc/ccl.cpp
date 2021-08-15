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
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include "oneflow/core/common/cplusplus_14.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/rank_group.h"

namespace py = pybind11;

namespace oneflow {

namespace {
Maybe<py::bytes> CpuBroadcast(py::bytes* in, size_t elem_cnt, int64_t root) {
  const auto& rank_group = JUST(RankGroup::DefaultRankGroup());
  const auto& parallel_desc = JUST(RankGroup::GetDefaultParallelDesc(DeviceType::kCPU, rank_group));
  JUST(ccl::Broadcast<DeviceType::kCPU>(&elem_cnt, &elem_cnt, sizeof(size_t), DataType::kChar, root,
                                        parallel_desc, nullptr));

  auto out = std::make_unique<char[]>(elem_cnt);
  if (GlobalProcessCtx::Rank() == root) {
    JUST(ccl::Broadcast<DeviceType::kCPU>(py::cast<std::string>(*in).c_str(), out.get(), elem_cnt,
                                          DataType::kChar, root, parallel_desc, nullptr));
  } else {
    JUST(ccl::Broadcast<DeviceType::kCPU>(nullptr, out.get(), elem_cnt, DataType::kChar, root,
                                          parallel_desc, nullptr));
  }
  return py::bytes(out.get(), elem_cnt);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("cpu_broadcast", [](py::bytes in, size_t elem_cnt, int64_t root) -> py::bytes {
    return CpuBroadcast(&in, elem_cnt, root).GetOrThrow();
  });
  m.def("cpu_broadcast", [](py::none in, size_t elem_cnt, int64_t root) -> py::bytes {
    return CpuBroadcast(nullptr, elem_cnt, root).GetOrThrow();
  });
}

}  // namespace oneflow
