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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/operator/op_node_signature.pb.h"
#include "oneflow/core/operator/op_node_signature.cfg.h"
#include "oneflow/core/job/sbp_parallel.pb.h"
#include "oneflow/core/job/mirrored_parallel.pb.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/mirrored_parallel.cfg.h"
#include "oneflow/core/register/batch_axis_signature.cfg.h"
#include "oneflow/core/job/parallel_signature.cfg.h"
#include "oneflow/core/common/data_type.cfg.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/register/blob_desc.cfg.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/eager/eager_symbol.pb.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"

namespace py = pybind11;

namespace oneflow {

namespace {

Maybe<cfg::OpNodeSignature> MakeOpNodeSignatureFromSerializedOpAttribute(
    const std::string& op_attribute_str) {
  OpAttribute op_attribute;
  CHECK_OR_RETURN(TxtString2PbMessage(op_attribute_str, &op_attribute))
      << "op_attribute parse failed";
  auto op_node_signature = std::make_shared<cfg::OpNodeSignature>();
  op_node_signature->mutable_sbp_signature()->InitFromProto(op_attribute.sbp_signature());
  op_node_signature->mutable_mirrored_signature()->InitFromProto(op_attribute.mirrored_signature());
  op_node_signature->mutable_logical_blob_desc_signature()->InitFromProto(
      op_attribute.logical_blob_desc_signature());
  op_node_signature->mutable_batch_axis_signature()->InitFromProto(
      op_attribute.batch_axis_signature());
  op_node_signature->mutable_parallel_signature()->InitFromProto(op_attribute.parallel_signature());
  return op_node_signature;
}

Maybe<eager::cfg::EagerSymbol> MakeEagerSymbol(const std::string& serialized_str) {
  eager::EagerSymbol eager_symbol;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_str, &eager_symbol))
      << "eager_symbol parse failed";
  return std::make_shared<eager::cfg::EagerSymbol>(eager_symbol);
}

}  // namespace

ONEFLOW_API_PYBIND11_MODULE("deprecated", m) {
  m.def("MakeOpNodeSignatureFromSerializedOpAttribute", [](const std::string& str) {
    return MakeOpNodeSignatureFromSerializedOpAttribute(str).GetPtrOrThrow();
  });

  m.def("MakeEagerSymbolByString",
        [](const std::string& str) { return MakeEagerSymbol(str).GetPtrOrThrow(); });
}

}  // namespace oneflow
