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
#include <pybind11/operators.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace py = pybind11;

namespace oneflow {

static const int64_t kMaxSplitAxis = 6;

namespace {

std::string SbpParallelSymbolToString(const Symbol<cfg::SbpParallel>& sbp_sym) {
  std::string sbp_str = "oneflow.sbp.";
  if (sbp_sym->has_broadcast_parallel()) {
    sbp_str += "broadcast";
  } else if (sbp_sym->has_partial_sum_parallel()) {
    sbp_str += "partial_sum";
  } else if (sbp_sym->has_split_parallel()) {
    sbp_str += "split(axis=" + std::to_string(sbp_sym->split_parallel().axis()) + ")";
  } else {
    UNIMPLEMENTED();
  }
  return sbp_str;
}

Maybe<std::vector<Symbol<cfg::SbpParallel>>> MakeSplitSbpParallelList(int max_split_axis) {
  std::shared_ptr<std::vector<Symbol<cfg::SbpParallel>>> ret =
      std::make_shared<std::vector<Symbol<cfg::SbpParallel>>>(max_split_axis);
  for (int i = 0; i < max_split_axis; ++i) {
    cfg::SbpParallel split_sbp_parallel;
    split_sbp_parallel.mutable_split_parallel()->set_axis(i);
    ret->at(i) = SymbolOf(split_sbp_parallel);
  }
  return ret;
}

Maybe<Symbol<cfg::SbpParallel>> MakeBroadcastSbpParallel() {
  cfg::SbpParallel broadcast_sbp;
  broadcast_sbp.mutable_broadcast_parallel();
  return SymbolOf(broadcast_sbp);
}

Maybe<Symbol<cfg::SbpParallel>> MakePartialSumSbpParallel() {
  cfg::SbpParallel partial_sum_sbp;
  partial_sum_sbp.mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_sbp);
}

Maybe<Symbol<cfg::SbpParallel>> GetSplitSbpParallel(int axis) {
  CHECK_LT_OR_RETURN(axis, kMaxSplitAxis);
  static std::vector<Symbol<cfg::SbpParallel>> split_sbp_sym_list =
      *JUST(MakeSplitSbpParallelList(kMaxSplitAxis));
  return split_sbp_sym_list.at(axis);
}

Maybe<Symbol<cfg::SbpParallel>> GetBroadcastSbpParallel() {
  static Symbol<cfg::SbpParallel> broadcast_sbp = JUST(MakeBroadcastSbpParallel());
  return broadcast_sbp;
}

Maybe<Symbol<cfg::SbpParallel>> GetPartialSumSbpParallel() {
  static Symbol<cfg::SbpParallel> partial_sum_sbp = JUST(MakePartialSumSbpParallel());
  return partial_sum_sbp;
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("sbp", m) {
  m.attr("max_split_axis") = kMaxSplitAxis;
  py::class_<Symbol<cfg::SbpParallel>, std::shared_ptr<Symbol<cfg::SbpParallel>>>(m, "sbp")
      .def("__str__", &SbpParallelSymbolToString)
      .def("__repr__", &SbpParallelSymbolToString)
      .def(py::self == py::self)
      .def(py::hash(py::self));
  m.def(
      "split", [](int axis) { return GetSplitSbpParallel(axis).GetOrThrow(); }, py::arg("axis"));
  m.def("broadcast", []() { return GetBroadcastSbpParallel().GetOrThrow(); });
  m.def("partial_sum", []() { return GetPartialSumSbpParallel().GetOrThrow(); });
}

}  // namespace oneflow
