#include "oneflow/core/operator/reshape_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape, const ShapeProto& reshape_proto) {
  size_t total_elem_dim_exclude_minus_1 = 1;
  bool has_minus_1 = false;
  bool minus_1_axis = -1;
  std::vector<int64_t> dim_vec;
  FOR_RANGE(int, axis, 0, reshape_proto.dim_size()) {
    int64_t dim = reshape_proto.dim(axis);
    dim_vec.push_back(dim);
    if (dim == -1) {
      OF_CHECK(has_minus_1 == false) << "only one `-1' supported";
      has_minus_1 = true;
      minus_1_axis = axis;
    } else if (dim > 0) {
      OF_CHECK_LE(dim, in_shape.elem_cnt()) << "invalid axis: " << axis << ", dim: " << dim;
      total_elem_dim_exclude_minus_1 *= dim;
      OF_CHECK_LE(total_elem_dim_exclude_minus_1, in_shape.elem_cnt())
          << "element number in reshape_conf is bigger than input blob";
    } else {
      OF_UNIMPLEMENTED() << "only positive number or -1 supported";
    }
  }
  OF_CHECK_EQ(in_shape.elem_cnt() % total_elem_dim_exclude_minus_1, 0);
  if (has_minus_1) {
    dim_vec[minus_1_axis] = in_shape.elem_cnt() / total_elem_dim_exclude_minus_1;
  } else {
    OF_CHECK_EQ(in_shape.elem_cnt(), total_elem_dim_exclude_minus_1)
        << "input blob's element number not equals reshape_conf";
  }
  return std::make_shared<Shape>(dim_vec);
}

Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                    HashMap<int, int>* squeezed_axis2origin_axis) {
  OF_CHECK_GT(origin.NumAxes(), 0);
  std::vector<int64_t> dim_vec;
  FOR_RANGE(int, axis, 0, origin.NumAxes()) {
    int64_t dim = origin.At(axis);
    OF_CHECK_GT(dim, 0);
    if (dim == 1) { continue; }
    OF_CHECK(squeezed_axis2origin_axis->emplace(dim_vec.size(), axis).second);
    dim_vec.push_back(dim);
  }
  *shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                        HashMap<int, int>* group_start_in_axis2out_axis) {
  CHECK_NE_OR_RETURN(in_shape.NumAxes(), 0);
  CHECK_NE_OR_RETURN(out_shape.NumAxes(), 0);
  OF_CHECK_EQ(in_shape.elem_cnt(), out_shape.elem_cnt());
  int in_axis = in_shape.NumAxes() - 1;
  int out_axis = out_shape.NumAxes() - 1;
  while (in_axis >= 0 && out_axis >= 0) {
    if (in_shape.Count(in_axis) < out_shape.Count(out_axis)) {
      --in_axis;
    } else if (in_shape.Count(in_axis) > out_shape.Count(out_axis)) {
      --out_axis;
    } else {
      if (in_shape.At(in_axis) == out_shape.At(out_axis)) {
        (*group_start_in_axis2out_axis)[in_axis] = out_axis;
      }
      --in_axis;
      --out_axis;
    }
  }
  const bool is_supported = in_axis + out_axis == -1 || in_axis + out_axis == -2;
  OF_CHECK(is_supported);
  return Maybe<void>::Ok();
}

}  // namespace

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

const PbMessage& ReshapeOp::GetCustomizedConf() const { return op_conf().reshape_conf(); }

Maybe<void> ReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;

  const ReshapeOpConf& conf = op_conf().reshape_conf();
  CHECK_GE_OR_RETURN(conf.shape().dim_size(), 1);
  std::vector<int64_t> dim_vec = {conf.shape().dim().begin(), conf.shape().dim().end()};
  int32_t dim_cnt_need_infer = 0;
  int64_t dim_index_need_infer = -1;
  int64_t elem_cnt = 1;

  for (int32_t i = 0; i < dim_vec.size(); ++i) {
    if (dim_vec[i] == -1) {
      ++dim_cnt_need_infer;
      dim_index_need_infer = i;
    }
  }
  CHECK_LE_OR_RETURN(dim_cnt_need_infer, 1);

  const auto& sbp_parallel_it = sbp_signature->bn_in_op2sbp_parallel().find("out");
  CHECK_OR_RETURN(sbp_parallel_it != sbp_signature->bn_in_op2sbp_parallel().end());
  const SbpParallel& sbp_parallel = sbp_parallel_it->second;
  if (sbp_parallel.has_split_parallel()) {
    const int64_t split_axis = sbp_parallel.split_parallel().axis();
    if (dim_index_need_infer != split_axis) {
      BalancedSplitter splitter(conf.shape().dim().Get(split_axis), parallel_ctx->parallel_num());
      CHECK_GE_OR_RETURN(conf.shape().dim().Get(split_axis), parallel_ctx->parallel_num());
      dim_vec[split_axis] = splitter.At(parallel_ctx->parallel_id()).size();
    }
  }

  if (dim_cnt_need_infer == 1) {
    for (int32_t i = 0; i < dim_vec.size(); ++i) {
      if (dim_vec[i] != -1) {
        CHECK_GT_OR_RETURN(dim_vec[i], 0);
        elem_cnt *= dim_vec[i];
      }
    }
    dim_vec[dim_index_need_infer] = in_blob_desc->shape().elem_cnt() / elem_cnt;
  }

  out_blob_desc->mut_shape() = Shape(dim_vec);
  CHECK_EQ_OR_RETURN(out_blob_desc->shape().elem_cnt(), in_blob_desc->shape().elem_cnt());
  return Maybe<void>::Ok();
}

Maybe<void> ReshapeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  HashMap<int, int> squeezed_group_start_in_axis2out_axis;
  HashMap<int, int> in_squeezed_axis2original_axis;
  HashMap<int, int> out_squeezed_axis2original_axis;
  {
    const auto& in_shape = JUST(LogicalBlobDesc4Ibn("in"))->shape();
    const auto& out_shape =
        JUST(GetLogicalOutBlobShape(in_shape, op_conf().reshape_conf().shape()));
    Shape squeezed_in_shape;
    Shape squeezed_out_shape;
    Squeeze(in_shape, &squeezed_in_shape, &in_squeezed_axis2original_axis);
    Squeeze(*out_shape, &squeezed_out_shape, &out_squeezed_axis2original_axis);
    GetGroupStartInAxis2OutAxis(squeezed_in_shape, squeezed_out_shape,
                                &squeezed_group_start_in_axis2out_axis);
  }
  for (const auto& pair : squeezed_group_start_in_axis2out_axis) {
    int64_t start_in_axis = in_squeezed_axis2original_axis.at(pair.first);
    int64_t start_out_axis = out_squeezed_axis2original_axis.at(pair.second);
    SbpSignatureBuilder()
        .Split(input_bns(), start_in_axis)
        .Split(output_bns(), start_out_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
