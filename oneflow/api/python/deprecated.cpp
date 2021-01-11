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
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob_desc.cfg.h"
#include "oneflow/core/register/blob_desc.pb.h"
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

Maybe<cfg::OptInt64> MakeOptInt64(const std::string& serialized_str) {
  OptInt64 opt_int64;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &opt_int64)) << "opt_int64 parse failed";
  return std::make_shared<cfg::OptInt64>(opt_int64);
}

Maybe<cfg::BlobDescProto> MakeBlobDescProto(const std::string& serialized_str) {
  BlobDescProto blob_desc;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &blob_desc)) << "blob_desc parse failed";
  return std::make_shared<cfg::BlobDescProto>(blob_desc);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  m.def("MakeSbpParrallelByString",
        [](const std::string& str) { return MakeSbpParallel(str).GetPtrOrThrow(); });

  m.def("MakeOptMirroredParrallelByString",
        [](const std::string& str) { return MakeOptMirroredParallel(str).GetPtrOrThrow(); });

  m.def("MakeOptInt64ByString",
        [](const std::string& str) { return MakeOptInt64(str).GetPtrOrThrow(); });

  m.def("MakeBlobDescProtoByString",
        [](const std::string& str) { return MakeBlobDescProto(str).GetPtrOrThrow(); });
}

}  // namespace oneflow
