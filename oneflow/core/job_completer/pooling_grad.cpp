#include "oneflow/core/job_completer/autograd.h"

namespace oneflow {

namespace {

PoolingConf PoolingConfFromPoolingOpConf(const OperatorConf& op_conf) {
  int32_t num_spatial_dims = 0;
  std::string pool_mode = "";
  const PbMessage* msg = nullptr;
  if (op_conf.has_average_pooling_1d_conf()) {
    num_spatial_dims = 1;
    pool_mode = "avg";
    msg = &op_conf.average_pooling_1d_conf();
  } else if (op_conf.has_average_pooling_2d_conf()) {
    num_spatial_dims = 2;
    pool_mode = "avg";
    msg = &op_conf.average_pooling_2d_conf();
  } else if (op_conf.has_average_pooling_3d_conf()) {
    num_spatial_dims = 3;
    pool_mode = "avg";
    msg = &op_conf.average_pooling_3d_conf();
  } else if (op_conf.has_max_pooling_1d_conf()) {
    num_spatial_dims = 1;
    pool_mode = "max";
    msg = &op_conf.max_pooling_1d_conf();
  } else if (op_conf.has_max_pooling_2d_conf()) {
    num_spatial_dims = 2;
    pool_mode = "max";
    msg = &op_conf.max_pooling_2d_conf();
  } else if (op_conf.has_max_pooling_3d_conf()) {
    num_spatial_dims = 3;
    pool_mode = "max";
    msg = &op_conf.max_pooling_3d_conf();
  } else {
    UNIMPLEMENTED();
  }
  PoolingConf pooling_conf;
  pooling_conf.set_num_spatial_dims(num_spatial_dims);
  pooling_conf.set_pool_mode(pool_mode);
  pooling_conf.set_data_format(GetValFromPbMessage<std::string>(*msg, "data_format"));
  pooling_conf.set_padding(GetValFromPbMessage<std::string>(*msg, "padding"));
  *pooling_conf.mutable_pool_size() = GetPbRfFromPbMessage<int32_t>(*msg, "pool_size");
  *pooling_conf.mutable_strides() = GetPbRfFromPbMessage<int32_t>(*msg, "strides");
  return pooling_conf;
}

void GenerateBackwardOpConf(
    const Operator& op, std::vector<OperatorConf>* op_confs,
    const std::function<LogicalBlobId*(const std::string&)>& DiffLbi4BnInOp) {
  CHECK(op.op_conf().has_average_pooling_1d_conf() || op.op_conf().has_average_pooling_2d_conf()
        || op.op_conf().has_average_pooling_3d_conf() || op.op_conf().has_max_pooling_1d_conf()
        || op.op_conf().has_max_pooling_2d_conf() || op.op_conf().has_max_pooling_3d_conf());
  const PoolingConf pooling_conf = PoolingConfFromPoolingOpConf(op.op_conf());
  LogicalBlobId* in_diff_lbi = DiffLbi4BnInOp("in");
  if (in_diff_lbi != nullptr) {
    OperatorConf pooling_grad_op;
    pooling_grad_op.set_name("System-AutoGrad-" + op.op_name() + "-Grad");
    PoolingGradOpConf* conf = pooling_grad_op.mutable_pooling_grad_conf();
    *conf->mutable_pooling_conf() = pooling_conf;
    conf->set_x(GenLogicalBlobName(op.BnInOp2Lbi("in")));
    conf->set_y(GenLogicalBlobName(op.BnInOp2Lbi("out")));
    conf->set_dy(GenLogicalBlobName(*DiffLbi4BnInOp("out")));
    conf->set_dx("dx");
    op_confs->push_back(pooling_grad_op);
    in_diff_lbi->set_op_name(pooling_grad_op.name());
    in_diff_lbi->set_blob_name(conf->dx());
  }
}

}  // namespace

REGISTER_OP_GRAD(OperatorConf::kAveragePooling1DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kAveragePooling2DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kAveragePooling3DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kMaxPooling1DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kMaxPooling2DConf, &GenerateBackwardOpConf);
REGISTER_OP_GRAD(OperatorConf::kMaxPooling3DConf, &GenerateBackwardOpConf);

}  // namespace oneflow