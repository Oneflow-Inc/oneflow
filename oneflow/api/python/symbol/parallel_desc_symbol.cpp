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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/placement.cfg.h"

namespace py = pybind11;

namespace oneflow {

Maybe<ParallelDesc> CreatePlacementSymbol(int64_t symbol_id,
                                          const std::shared_ptr<cfg::ParallelConf>& symbol_conf) {
  ParallelConf symbol_pb;
  symbol_conf->ToProto(&symbol_pb);
  return ParallelDesc::New(symbol_id, symbol_pb);
}

std::pair<std::shared_ptr<ParallelDesc>, std::shared_ptr<cfg::ErrorProto>> ApiCreatePlacementSymbol(
    int64_t symbol_id, const std::shared_ptr<cfg::ParallelConf>& symbol_conf) {
  return CreatePlacementSymbol(symbol_id, symbol_conf).GetDataPtrAndErrorProto();
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<ParallelDesc, std::shared_ptr<ParallelDesc>>(m, "ParallelDesc")
      .def_property_readonly("symbol_id",
                             [](std::shared_ptr<ParallelDesc> parallel_desc) {
                               const auto& symbol_id = CHECK_JUST(parallel_desc->symbol_id());
                               return symbol_id;
                             })
      .def_property_readonly(
          "parallel_conf",
          [](std::shared_ptr<ParallelDesc> parallel_desc) {
            return std::make_shared<cfg::ParallelConf>(parallel_desc->parallel_conf());
          })
      .def_property_readonly("parallel_num", &ParallelDesc::parallel_num)
      .def_property_readonly("device_tag", &ParallelDesc::device_tag)
      .def_property_readonly("machine_id2device_id_list",
                             &ParallelDesc::machine_id2sorted_dev_phy_ids)
      .def("Containing", &ParallelDesc::Bigger)
      .def(py::self == py::self)
      .def(py::hash(py::self));

  m.def("CreatePlacementSymbol", &ApiCreatePlacementSymbol);
}

}  // namespace oneflow
