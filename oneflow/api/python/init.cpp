#include <pybind11/pybind11.h>
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/job/global_for.h"

namespace py = pybind11;

namespace oneflow {

PYBIND11_MODULE(oneflow_internal2, m) {
  m.def("EagerExecutionEnabled", []() { return *Global<bool, EagerExecution>::Get(); },
        R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
}
}  // namespace oneflow
