#include "oneflow/core/kernel/redefine_dim0_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void RedefineDim0Kernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void RedefineDim0Kernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob(GenDiffBn("in"))->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(GenDiffBn("out")));
}

template<DeviceType device_type>
void RedefineDim0Kernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const RedefineDim0OpConf& conf = this->op_conf().redefine_dim0_conf();
  CHECK_EQ(conf.type_case(), RedefineDim0OpConf::kShrinkConf);
  CHECK_EQ(conf.shrink_conf().axis(), 1);
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  int64_t inst_num = out_blob->shape().At(0);
  CHECK_EQ(in_blob->dim0_inner_shape().At(0), inst_num);
  FOR_RANGE(int64_t, i, 0, inst_num) {
    out_blob->set_dim1_valid_num(i, in_blob->dim0_valid_num(i));
  }
}

template<DeviceType device_type>
void RedefineDim0Kernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const RedefineDim0OpConf& conf = this->op_conf().redefine_dim0_conf();
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  switch (conf.type_case()) {
    case RedefineDim0OpConf::kShrinkConf: {
      CHECK(in_blob->has_dim0_inner_shape());
      CHECK(out_blob->has_dim0_inner_shape());
      FOR_RANGE(int64_t, i, 0, out_blob->dim0_inner_shape().At(0)) {
        out_blob->set_dim0_valid_num(i, in_blob->dim0_valid_num(i));
      }
      break;
    }
    case RedefineDim0OpConf::kExtendConf: {
      bool has_inner_shape = in_blob->has_dim0_inner_shape();
      FOR_RANGE(int64_t, i, 0, out_blob->shape().At(0)) {
        int32_t valid_num =
            has_inner_shape ? in_blob->dim0_valid_num(i) : in_blob->dim1_valid_num(i);
        out_blob->set_dim0_valid_num(i, valid_num);
      }
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
}

template<DeviceType device_type>
void RedefineDim0Kernel<device_type>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const RedefineDim0OpConf& conf = this->op_conf().redefine_dim0_conf();
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  switch (conf.type_case()) {
    case RedefineDim0OpConf::kShrinkConf: {
      CHECK(out_diff_blob->has_dim0_inner_shape());
      CHECK(in_diff_blob->has_dim0_inner_shape());
      FOR_RANGE(int64_t, i, 0, in_diff_blob->dim0_inner_shape().At(0)) {
        in_diff_blob->set_dim0_valid_num(i, out_diff_blob->dim0_valid_num(i));
      }
      break;
    }
    case RedefineDim0OpConf::kExtendConf: {
      CHECK(out_diff_blob->has_dim0_inner_shape());
      if (in_diff_blob->has_dim0_inner_shape()) {
        FOR_RANGE(int64_t, i, 0, in_diff_blob->dim0_inner_shape().At(0)) {
          in_diff_blob->set_dim0_valid_num(i, out_diff_blob->dim0_valid_num(i));
        }
      } else {
        FOR_RANGE(int64_t, i, 0, out_diff_blob->dim0_inner_shape().At(0)) {
          in_diff_blob->set_dim1_valid_num(i, out_diff_blob->dim0_valid_num(i));
        }
      }
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kRedefineDim0Conf, RedefineDim0Kernel);

}  // namespace oneflow
