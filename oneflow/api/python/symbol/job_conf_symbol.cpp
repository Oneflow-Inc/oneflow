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
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_conf.cfg.h"

namespace py = pybind11;

namespace oneflow {

Maybe<JobDesc> CreateJobConfSymbol(int64_t symbol_id,
                                   const std::shared_ptr<cfg::JobConfigProto>& symbol_conf) {
  JobConfigProto symbol_pb;
  symbol_conf->ToProto(&symbol_pb);
  return JobDesc::New(symbol_id, symbol_pb);
}

ONEFLOW_API_PYBIND11_MODULE("", m) {
  py::class_<JobDesc, std::shared_ptr<JobDesc>>(m, "JobConfSymbol")
      .def(py::init([](int64_t symbol_id, const std::shared_ptr<cfg::JobConfigProto>& symbol_conf) {
        return CreateJobConfSymbol(symbol_id, symbol_conf).GetPtrOrThrow();
      }))
      .def_property_readonly("symbol_id",
                             [](const JobDesc& x) { return x.symbol_id().GetOrThrow(); })
      .def_property_readonly("data", &JobDesc::cfg_job_conf);
}

}  // namespace oneflow
