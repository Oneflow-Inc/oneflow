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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/global_mode.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("global_view", m) {
  py::class_<GlobalMode::Guard, std::shared_ptr<GlobalMode::Guard>>(m, "global_mode")
      .def(py::init([](const bool enabled) {
        if (enabled) {
          THROW(RuntimeError) << "To enable global mode, placement and sbp must be provided.";
        }
        return std::make_shared<GlobalMode::Guard>(enabled);
      }))
      .def(py::init([](const bool enabled, const Symbol<ParallelDesc>& placement,
                       const std::vector<Symbol<SbpParallel>>& sbp) {
             if (!enabled) {
               THROW(RuntimeError)
                   << "To disable global mode, placement and sbp must not be provided.";
             }
             return std::make_shared<GlobalMode::Guard>(enabled, CHECK_JUST(GetNdSbp(sbp)),
                                                        placement);
           }),
           py::arg("enabled").none(false), py::arg("placement").none(false),
           py::arg("sbp").none(false))
      .def(py::init([](const bool enabled, const Symbol<ParallelDesc>& placement,
                       const Symbol<SbpParallel>& sbp) {
             return std::make_shared<GlobalMode::Guard>(enabled, CHECK_JUST(SbpToNdSbp(sbp)),
                                                        placement);
           }),
           py::arg("enabled").none(false), py::arg("placement").none(false),
           py::arg("sbp").none(false))
      .def("__enter__", [](const GlobalMode::Guard& guard_obj) {})
      .def("__exit__", [](const GlobalMode::Guard& guard_obj, const py::object& type,
                          const py::object& value, const py::object& traceback) {});

  py::class_<GlobalMode, std::shared_ptr<GlobalMode>>(m, "current_global_mode")
      .def(py::init([]() { return std::make_shared<GlobalMode>(); }))
      .def_property_readonly("is_enabled", [](const GlobalMode& gm) { return gm.is_enabled(); })
      .def_property_readonly("sbp",
                             [](const GlobalMode& gm) {
                               if (!gm.is_enabled()) {
                                 THROW(RuntimeError)
                                     << "Current global mode is disabled, there is no sbp.";
                               }
                               const auto& nd_sbp = gm.nd_sbp();
                               auto tuple = py::tuple(nd_sbp->sbp_parallel_size());
                               for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
                                 tuple[i] = SymbolOf(nd_sbp->sbp_parallel(i));
                               }
                               return tuple;
                             })
      .def_property_readonly("placement", [](const GlobalMode& gm) {
        if (!gm.is_enabled()) {
          THROW(RuntimeError) << "Current global mode is disabled, there is no placement.";
        }
        return gm.parallel_desc();
      });
}

}  // namespace oneflow
