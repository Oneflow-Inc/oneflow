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
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_parallel.cfg.h"
#include "oneflow/core/job/mirrored_parallel.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/maybe.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<cfg::SbpParallel> MakeSbpParallel(const std::string& serialized_str) {
  SbpParallel sbp_parallel;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &sbp_parallel))
      << "sbp_parallel parse failed";
  return std::make_shared<cfg::SbpParallel>(sbp_parallel);
}

Maybe<cfg::OptMirroredParallel> MakeOptMirroredParallel(const std::string& serialized_str) {
  OptMirroredParallel opt_mirrored_parallel;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &opt_mirrored_parallel))
      << "opt_mirrored_parallel parse failed";
  return std::make_shared<cfg::OptMirroredParallel>(opt_mirrored_parallel);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  m.def("MakeSbpParrallelByString",
        [](const std::string& str) { return MakeSbpParallel(str).GetPtrOrThrow(); });

  m.def("MakeOptMirroredParrallelByString",
        [](const std::string& str) { return MakeOptMirroredParallel(str).GetPtrOrThrow(); });
}

}  // namespace oneflow
