#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include "oneflow/api/common/folder_rule_table.h"
#include "oneflow/api/python/of_api_registry.h"

namespace py = pybind11;

namespace oneflow {

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("GetFolderRuleTable", &GetFolderRuleTable, py::return_value_policy::reference_internal);
}

}  // namespace oneflow
