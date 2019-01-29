#include "oneflow/core/operator/gather_op.h"

namespace oneflow {

namespace {

int64_t GetGatherAxis(const GatherOpConf& conf, int64_t num_axes) {
  const int64_t axis = conf.axis() < 0 ? num_axes + conf.axis() : conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

int64_t GetGatherAxis(const GatherOpConf& conf, const BlobDesc* in_blob_desc) {
  return GetGatherAxis(conf, in_blob_desc->shape().NumAxes());
}

}  // namespace

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

bool GatherOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return ibn == "in";
}

void GatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), in);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  std::vector<int64_t> dim_vec;
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                 in->shape().dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                 indices->shape().dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                 in->shape().dim_vec().end());
  out->mut_shape() = Shape(dim_vec);
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_gather_conf()->set_axis(axis);
}

void GatherOp::InferOutputBlobModelSplitAxis(
    std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  CHECK_EQ(*ModelSplitAxis4BnInOp("indices"), -1);
  const int32_t in_model_split_axis = *ModelSplitAxis4BnInOp("in");
  const int64_t in_num_axes = ShapeNumAxes4BnInOp("in");
  const int64_t gather_axis = GetGatherAxis(op_conf().gather_conf(), in_num_axes);
  if (in_model_split_axis != -1) {
    CHECK_GT(in_model_split_axis, gather_axis);
    CHECK_LT(in_model_split_axis, in_num_axes);
    *ModelSplitAxis4BnInOp("out") = in_model_split_axis + ShapeNumAxes4BnInOp("indices") - 1;
  } else {
    CHECK_EQ(parallel_context->policy(), ParallelPolicy::kDataParallel);
    *ModelSplitAxis4BnInOp("out") = -1;
  }
}
/*
void GatherOp::InferInputOutputLogicalBlobParallelDesc(
    std::function<LogicalBlobParallelDesc*(const std::string&)> LogicalBlobParallelDesc4BnInOp,
    std::function<const LogicalBlobParallelDesc&(const std::string&)>
ProducerLogicalBlobParallelDesc4Ibn, std::function<int32_t(const std::string&)>
ModelSplitAxis4BnInOp, const ParallelContext* parallel_ctx) const { const auto& producer_in_blob_pt
= ProducerLogicalBlobParallelDesc4Ibn("in"); const auto& producer_indices_blob_pt =
ProducerLogicalBlobParallelDesc4Ibn("indices"); if (parallel_ctx->policy() == kDataParallel) {
    CHECK(producer_in_blob_pt->has_clone_parallel());
    CHECK(producer_indices_blob_pt->has_split_parallel());
    CHECK_EQ(producer_indices_blob_pt->split_parallel().axis(), 0);
    LogicalBlobParallelDesc4BnInOp("in")->mutable_clone_parallel();
    LogicalBlobParallelDesc4BnInOp("indices")->mutable_split_parallel()->set_axis(0);
    LogicalBlobParallelDesc4BnInOp("out")->mutable_split_parallel()->set_axis(0);
  } else if (parallel_ctx->policy() == kModelParallel) {
    CHECK(producer_indices_blob_pt->has_clone_parallel());
    LogicalBlobParallelDesc4BnInOp("indices")->mutable_clone_parallel();
    CHECK(producer_in_blob_pt->has_split_parallel());
    int32_t model_split_axis = ModelSplitAxis4BnInOp("in");
    CHECK_EQ(producer_in_blob_pt.split_parallel().axis(), model_split_axis);
    LogicalBlobParallelDesc4BnInOp("in")->mutable_split_parallel()->set_axis(model_split_axis);
    if (model_split_axis == 0) {
      LogicalBlobParallelDesc4BnInOp("out")->mutable_partial_sum_parallel();
    } else if (model_split_axis == 1) {
      int32_t out_split_axis = ModelSplitAxis4BnInOp("out");
      CHECK_NE(out_split_axis, -1);
      CHECK_NE(out_split_axis, 0);
      LogicalBlobParallelDesc4BnInOp("out")->mutable_split_parallel()->set_axis(out_split_axis);
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UNIMPLEMENTED();
  }
}
*/
REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
