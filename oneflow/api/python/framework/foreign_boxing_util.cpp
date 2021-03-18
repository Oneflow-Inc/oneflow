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
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/eager/foreign_boxing_util.h"

namespace py = pybind11;
namespace oneflow {

class PyForeignBoxingUtil : public ForeignBoxingUtil {
 public:
  using ForeignBoxingUtil::ForeignBoxingUtil;

  std::shared_ptr<compatible_py::BlobObject> BoxingTo(
      const std::shared_ptr<InstructionsBuilder>& builder,
      const std::shared_ptr<compatible_py::BlobObject>& blob_object,
      const std::shared_ptr<compatible_py::OpArgParallelAttribute>& op_arg_parallel_attr)
      const override {
    PYBIND11_OVERRIDE(std::shared_ptr<compatible_py::BlobObject>, ForeignBoxingUtil, BoxingTo,
                      builder, blob_object, op_arg_parallel_attr);
  }

  std::shared_ptr<ParallelDesc> TryReplaceDeviceTag(
      const std::shared_ptr<InstructionsBuilder>& builder,
      const std::shared_ptr<ParallelDesc>& parallel_desc_symbol,
      const std::string& device_tag) const override {
    PYBIND11_OVERRIDE(std::shared_ptr<ParallelDesc>, ForeignBoxingUtil, TryReplaceDeviceTag,
                      builder, parallel_desc_symbol, device_tag);
  }

  void Assign(const std::shared_ptr<InstructionsBuilder>& builder,
              const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
              const std::shared_ptr<compatible_py::BlobObject>& source_blob_object) const override {
    PYBIND11_OVERRIDE(void, ForeignBoxingUtil, Assign, builder, target_blob_object,
                      source_blob_object);
  }
};

Maybe<void> RegisterBoxingUtilOnlyOnce(const std::shared_ptr<ForeignBoxingUtil>& boxing_util) {
  CHECK_ISNULL_OR_RETURN(Global<std::shared_ptr<ForeignBoxingUtil>>::Get())
      << "Foreign Boxing util has been registered.";
  Global<std::shared_ptr<ForeignBoxingUtil>>::New(boxing_util);
  return Maybe<void>::Ok();
}

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  using namespace oneflow;
  py::class_<ForeignBoxingUtil, PyForeignBoxingUtil, std::shared_ptr<ForeignBoxingUtil>>(
      m, "ForeignBoxingUtil")
      .def(py::init<>())
      .def("BoxingTo", &ForeignBoxingUtil::BoxingTo)
      .def("TryReplaceDeviceTag", &ForeignBoxingUtil::TryReplaceDeviceTag)
      .def("Assign", &ForeignBoxingUtil::Assign);

  m.def("RegisterBoxingUtilOnlyOnce",
        [](const std::shared_ptr<oneflow::ForeignBoxingUtil>& boxing_util) {
          oneflow::RegisterBoxingUtilOnlyOnce(boxing_util).GetOrThrow();
        });
}

}  // namespace oneflow
