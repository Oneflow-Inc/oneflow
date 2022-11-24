
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/job/global_mode.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("global_mode", m) {
  py::class_<GlobalMode::Guard, std::shared_ptr<GlobalMode::Guard>>(m, "guard")
      .def(py::init(
          [](const bool is_enabled) { return std::make_shared<GlobalMode::Guard>(is_enabled); }))
      .def(py::init(
          [](const bool is_enabled, const Symbol<ParallelDesc>& placement, const std::vector<Symbol<SbpParallel>>& sbp) { 
            return std::make_shared<GlobalMode::Guard>(is_enabled, CHECK_JUST(GetNdSbp(sbp)), placement); }),
            py::arg("is_enabled").none(false), py::arg("placement").none(false), py::arg("sbp").none(false))
      .def(py::init(
          [](const bool is_enabled, const Symbol<ParallelDesc>& placement, const Symbol<SbpParallel>& sbp) { 
            return std::make_shared<GlobalMode::Guard>(is_enabled, CHECK_JUST(SbpToNdSbp(sbp)), placement); }),
            py::arg("is_enabled").none(false), py::arg("placement").none(false), py::arg("sbp").none(false))
      .def("__enter__", [](const GlobalMode::Guard& guard_obj) {})
      .def("__exit__", [](const GlobalMode::Guard& guard_obj, const py::object& type,
                          const py::object& value, const py::object& traceback) {});

  m.def("is_enabled", []() { return GlobalMode::is_enabled(); });
  m.def("sbp", []() {
  const auto& nd_sbp = GlobalMode::nd_sbp();
  auto tuple = py::tuple(nd_sbp->sbp_parallel_size());
  for (int i = 0; i < nd_sbp->sbp_parallel_size(); ++i) {
    tuple[i] = SymbolOf(nd_sbp->sbp_parallel(i));
  }
  return tuple;
 });
  m.def("placement", []() { return GlobalMode::parallel_desc(); });
}

}  // namespace oneflow
