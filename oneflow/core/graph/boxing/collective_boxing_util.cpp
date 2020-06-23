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

}  // namespace

bool GenericOpHasInput(const RankDesc& rank_desc) {
  const OpType op_type = rank_desc.op_desc().op_type();
  if (op_type == OpType::kOpTypeAllReduce || op_type == OpType::kOpTypeAllGather
      || op_type == OpType::kOpTypeReduceScatter || op_type == OpType::kOpTypeReduce) {
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
      || op_type == OpType::kOpTypeReduceScatter || op_type == OpType::kOpTypeBroadcast) {
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
  } else {
    UNIMPLEMENTED();
    return Shape();
  }
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
