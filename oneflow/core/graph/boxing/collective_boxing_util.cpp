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
#include "oneflow/core/graph/boxing/collective_boxing_util.h"

namespace oneflow {

namespace boxing {

namespace collective {

namespace {

Shape GetSplitShape(const RankDesc& rank_desc) {
  Shape shape(rank_desc.op_desc().shape());
  CHECK_GT(shape.NumAxes(), 0);
  CHECK(shape.At(0) % rank_desc.op_desc().num_ranks() == 0);
  shape.Set(0, shape.At(0) / rank_desc.op_desc().num_ranks());
  return shape;
}

Shape GetFlattenSplitShape(const RankDesc& rank_desc) {
  Shape shape(rank_desc.op_desc().shape());
  CHECK_GT(shape.NumAxes(), 0);
  CHECK(shape.elem_cnt() % rank_desc.op_desc().num_ranks() == 0);
  Shape return_shape({shape.elem_cnt() / rank_desc.op_desc().num_ranks()});
  return return_shape;
}

}  // namespace

bool GenericOpHasInput(const RankDesc& rank_desc) {
  const OpType op_type = rank_desc.op_desc().op_type();
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeAllGather
      || op_type == OpType::kOpTypeReduceScatter || op_type == OpType::kOpTypeReduce
      || op_type == OpType::kOpTypeAll2All) {
    return true;
  } else if (op_type == OpType::kOpTypeBroadcast) {
    CHECK(rank_desc.op_desc().has_root());
    return rank_desc.rank() == rank_desc.op_desc().root();
  } else {
    UNIMPLEMENTED();
    return false;
  }
}

bool GenericOpHasOutput(const RankDesc& rank_desc) {
  const OpType op_type = rank_desc.op_desc().op_type();
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeAllGather
      || op_type == OpType::kOpTypeReduceScatter || op_type == OpType::kOpTypeBroadcast
      || op_type == OpType::kOpTypeAll2All) {
    return true;
  } else if (op_type == OpType::kOpTypeReduce) {
    CHECK(rank_desc.op_desc().has_root());
    return rank_desc.rank() == rank_desc.op_desc().root();
  } else {
    UNIMPLEMENTED();
    return false;
  }
}

Shape GenericOpGetInputShape(const RankDesc& rank_desc) {
  CHECK(GenericOpHasInput(rank_desc));
  const OpType op_type = rank_desc.op_desc().op_type();
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeReduceScatter
      || op_type == OpType::kOpTypeReduce || op_type == OpType::kOpTypeBroadcast) {
    return Shape(rank_desc.op_desc().shape());
  } else if (op_type == OpType::kOpTypeAllGather) {
    return GetSplitShape(rank_desc);
  } else if (op_type == OpType::kOpTypeAll2All) {
    return GetFlattenSplitShape(rank_desc);
  } else {
    UNIMPLEMENTED();
    return Shape();
  }
}

Shape GenericOpGetOutputShape(const RankDesc& rank_desc) {
  CHECK(GenericOpHasOutput(rank_desc));
  const OpType op_type = rank_desc.op_desc().op_type();
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeAllGather
      || op_type == OpType::kOpTypeReduce || op_type == OpType::kOpTypeBroadcast) {
    return Shape(rank_desc.op_desc().shape());
  } else if (op_type == OpType::kOpTypeReduceScatter) {
    return GetSplitShape(rank_desc);
  } else if (op_type == OpType::kOpTypeAll2All) {
    return GetFlattenSplitShape(rank_desc);
  } else {
    UNIMPLEMENTED();
    return Shape();
  }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
