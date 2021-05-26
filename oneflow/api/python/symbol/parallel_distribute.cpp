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
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace py = pybind11;

namespace oneflow {

namespace {

std::string ParallelDistributionSymbolToString(const Symbol<cfg::ParallelDistribution>& x) {
  py::list sbp_parallels;
  std::stringstream ss;
  int32_t idx = 0;
  ss << "ParallelDistribution([";
  for (const auto& sbp_para : x->sbp_parallel()) {
    SbpParallel proto_sbp;
    sbp_para.ToProto(&proto_sbp);
    ss << SbpParallelToString(proto_sbp);
    if (++idx != x->sbp_parallel_size()) { ss << ", "; }
  }
  ss << "])";
  return ss.str();
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<Symbol<cfg::ParallelDistribution>, std::shared_ptr<Symbol<cfg::ParallelDistribution>>>(
      m, "ParallelDistributionSymbol")
      .def(py::init([](const std::vector<std::string>& sbp_parallels) {
        cfg::ParallelDistribution parallel_distribution;
        SbpParallel sbp_parallel;
        for (const std::string& sbp_parallel_str : sbp_parallels) {
          CHECK(ParseSbpParallelFromString(sbp_parallel_str, &sbp_parallel))
              << "invalid sbp_parallel: " << sbp_parallel_str;
          parallel_distribution.mutable_sbp_parallel()->Add()->InitFromProto(sbp_parallel);
        }
        return Symbol<cfg::ParallelDistribution>(parallel_distribution);
      }))
      .def("__str__", &ParallelDistributionSymbolToString)
      .def("__repr__", &ParallelDistributionSymbolToString);
}

}  // namespace oneflow
