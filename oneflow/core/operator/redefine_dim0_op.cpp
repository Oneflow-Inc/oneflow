#include "oneflow/core/operator/redefine_dim0_op.h"

namespace oneflow {

void RedefineDim0Op::InitFromOpConf() {
  CHECK(op_conf().has_redefine_dim0_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& RedefineDim0Op::GetCustomizedConf() const {
  return op_conf().redefine_dim0_conf();
}

void RedefineDim0Op::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  std::vector<int64_t> shape_vec;
  std::vector<int64_t> inner_shape_vec;
  const RedefineDim0OpConf& conf = op_conf().redefine_dim0_conf();
  switch (conf.type_case()) {
    case RedefineDim0OpConf::kShrinkConf: {
      int64_t shrink_axis = conf.shrink_conf().axis();
      CHECK_GE(shrink_axis, 1);
      CHECK(in_blob_desc->has_dim0_inner_shape());
      CHECK(!in_blob_desc->has_dim1_valid_num_field());
      const Shape& shape = in_blob_desc->shape();
      const Shape& inner_shape = in_blob_desc->dim0_inner_shape();
      inner_shape_vec.reserve(shrink_axis);
      FOR_RANGE(int64_t, axis, 0, shrink_axis) {
        inner_shape_vec.emplace_back(inner_shape.At(axis));
      }
      shape_vec.reserve(shape.dim_vec().size() + inner_shape.NumAxes() - shrink_axis);
      FOR_RANGE(int64_t, axis, shrink_axis, inner_shape.NumAxes()) {
        shape_vec.emplace_back(inner_shape.At(axis));
      }
      shape_vec.insert(shape_vec.end(), ++shape.dim_vec().begin(), shape.dim_vec().end());
      if (shrink_axis == 1) {
        out_blob_desc->set_has_dim1_valid_num_field(in_blob_desc->has_dim0_valid_num_field());
        out_blob_desc->set_has_dim0_valid_num_field(false);
      } else {
        out_blob_desc->set_has_dim1_valid_num_field(false);
        out_blob_desc->set_has_dim0_valid_num_field(in_blob_desc->has_dim0_valid_num_field());
      }
      break;
    }
    case RedefineDim0OpConf::kExtendConf: {
      int64_t extend_axis = conf.extend_conf().axis();
      CHECK_GE(extend_axis, 1);
      const Shape& shape = in_blob_desc->shape();
      bool has_inner_shape = false;
      if (in_blob_desc->has_dim0_inner_shape()) {
        CHECK(!in_blob_desc->has_dim1_valid_num_field());
        const Shape& inner_shape = in_blob_desc->dim0_inner_shape();
        inner_shape_vec.reserve(inner_shape.NumAxes() + extend_axis);
        inner_shape_vec.insert(inner_shape_vec.begin(), inner_shape.dim_vec().begin(),
                               inner_shape.dim_vec().end());
        has_inner_shape = true;
      } else {
        inner_shape_vec.reserve(1 + extend_axis);
        inner_shape_vec.emplace_back(shape.At(0));
      }
      int64_t extend_dims = 1;
      FOR_RANGE(int64_t, axis, 1, extend_axis + 1) {
        int64_t dim = shape.At(axis);
        inner_shape_vec.emplace_back(dim);
        extend_dims *= dim;
      }
      shape_vec.reserve(shape.dim_vec().size() - extend_axis);
      shape_vec.emplace_back(shape.At(0) * extend_dims);
      FOR_RANGE(int64_t, axis, extend_axis + 1, shape.NumAxes()) {
        shape_vec.emplace_back(shape.At(axis));
      }
      out_blob_desc->set_has_dim1_valid_num_field(false);
      if (has_inner_shape) {
        out_blob_desc->set_has_dim0_valid_num_field(in_blob_desc->has_dim0_valid_num_field());
      } else {
        out_blob_desc->set_has_dim0_valid_num_field(in_blob_desc->has_dim1_valid_num_field());
      }
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  out_blob_desc->mut_shape() = Shape(shape_vec);
  if (inner_shape_vec.size() > 1) {
    out_blob_desc->mut_dim0_inner_shape() = Shape(inner_shape_vec);
  }
}

REGISTER_OP(OperatorConf::kRedefineDim0Conf, RedefineDim0Op);

}  // namespace oneflow
