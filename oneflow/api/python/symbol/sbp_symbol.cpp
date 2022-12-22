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
#include "oneflow/api/common/sbp.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/nd_sbp.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<std::vector<Symbol<SbpParallel>>> MakeSplitSbpParallelList(int max_split_axis) {
  std::shared_ptr<std::vector<Symbol<SbpParallel>>> ret =
      std::make_shared<std::vector<Symbol<SbpParallel>>>(max_split_axis);
  for (int i = 0; i < max_split_axis; ++i) { ret->at(i) = JUST(MakeSplitSbpParallel(i)); }
  return ret;
}

Maybe<Symbol<SbpParallel>> GetSplitSbpParallel(int axis) {
  CHECK_GE_OR_RETURN(axis, 0) << Error::RuntimeError()
                              << "Split axis must not be negative, but got " << axis << "!";
  CHECK_LT_OR_RETURN(axis, kMaxSplitAxis)
      << Error::RuntimeError() << "Expected split axis to be less than the supported maximum axis ("
      << kMaxSplitAxis << "), but got " << axis << "!";
  static std::vector<Symbol<SbpParallel>> split_sbp_sym_list =
      *JUST(MakeSplitSbpParallelList(kMaxSplitAxis));
  return split_sbp_sym_list.at(axis);
}

Maybe<Symbol<SbpParallel>> GetBroadcastSbpParallel() {
  static Symbol<SbpParallel> broadcast_sbp = JUST(MakeBroadcastSbpParallel());
  return broadcast_sbp;
}

Maybe<Symbol<SbpParallel>> GetPartialSumSbpParallel() {
  static Symbol<SbpParallel> partial_sum_sbp = JUST(MakePartialSumSbpParallel());
  return partial_sum_sbp;
}

Maybe<std::pair<std::string, int>> SbpGetState(const Symbol<SbpParallel>& sbp) {
  if (sbp->has_broadcast_parallel()) {
    return std::make_shared<std::pair<std::string, int>>("B", -1);
  } else if (sbp->has_partial_sum_parallel()) {
    return std::make_shared<std::pair<std::string, int>>("P", -1);
  } else if (sbp->has_split_parallel()) {
    return std::make_shared<std::pair<std::string, int>>("S", sbp->split_parallel().axis());
  } else {
    return Error::RuntimeError() << "Invalid sbp signature: " << sbp->DebugString();
  }
}

Maybe<Symbol<SbpParallel>> GetSbpFromState(const std::pair<std::string, int>& state) {
  if (state.first == "B") {
    return GetBroadcastSbpParallel();
  } else if (state.first == "P") {
    return GetPartialSumSbpParallel();
  } else if (state.first == "S") {
    return GetSplitSbpParallel(state.second);
  } else {
    return Error::RuntimeError() << "Invalid sbp signature state: (" << state.first << ", "
                                 << state.second << ");";
  }
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("sbp", m) {
  m.attr("max_split_axis") = kMaxSplitAxis;
  py::class_<Symbol<SbpParallel>, std::shared_ptr<Symbol<SbpParallel>>>(m, "sbp",
                                                                        py::dynamic_attr())
      .def("__str__", &api::ApiSbpToString)
      .def("__repr__", &api::ApiSbpToString)
      .def(py::self == py::self)
      .def(py::hash(py::self))
      .def("_ToAttrStr",
           [](const Symbol<SbpParallel>& sbp_sym) { return SbpParallelToString(*sbp_sym); })
      .def(py::pickle(
          [](const Symbol<SbpParallel>& sbp) {  // __getstate__
            return SbpGetState(sbp).GetOrThrow();
          },
          [](const std::pair<std::string, int>& state) {  // __setstate__
            return GetSbpFromState(state).GetOrThrow();
          }));
  m.def("split", GetSplitSbpParallel, py::arg("axis"));
  m.def("broadcast", &GetBroadcastSbpParallel);
  m.def("partial_sum", &GetPartialSumSbpParallel);
}

}  // namespace oneflow
