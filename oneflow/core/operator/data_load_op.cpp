#include "oneflow/core/operator/data_load_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

int64_t GetDeviceBatchSize(const int64_t batch_size, const ParallelContext* parallel_ctx) {
  int64_t parallel_num = parallel_ctx->parallel_num();
  CHECK_GE(batch_size, parallel_num);
  CHECK_EQ(batch_size % parallel_num, 0);
  return batch_size / parallel_num;
}

}  // namespace

void DataLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_data_load_conf());
  const DataLoadOpConf& conf = op_conf().data_load_conf();
  if (conf.has_tick()) { EnrollInputBn("tick", false); }
  FOR_RANGE(int32_t, i, 0, conf.blobs_size()) {
    CHECK(conf.blobs(i).has_data_source());
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& DataLoadOp::GetCustomizedConf() const { return op_conf().data_load_conf(); }

LogicalBlobId DataLoadOp::obn2lbi(const std::string& output_bn) const {
  CHECK_STREQ(output_bn.substr(0, 4).c_str(), "out_");
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(
      op_conf().data_load_conf().blobs(oneflow_cast<int32_t>(output_bn.substr(4))).name());
  return ret;
}

void DataLoadOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const DataLoadOpConf& conf = op_conf().data_load_conf();
  DataLoadKernelConf* this_kernel_conf = kernel_conf->mutable_data_load_conf();
  // Set data_instance
  DataInstanceProto* data_inst = this_kernel_conf->mutable_data_instance();
  for (const BlobConf& blob_conf : conf.blobs()) {
    DataFieldProto* data_field = data_inst->add_data_fields();
    data_field->set_data_source(blob_conf.data_source());
    *data_field->mutable_shape() = blob_conf.shape();
    data_field->set_dtype(blob_conf.data_type());
  }
  // Set parallel info
  int64_t device_batch_size = GetDeviceBatchSize(conf.batch_size(), parallel_ctx);
  this_kernel_conf->set_device_batch_size(device_batch_size);
  this_kernel_conf->set_parallel_num(parallel_ctx->parallel_num());
  this_kernel_conf->set_parallel_id(parallel_ctx->parallel_id());
}

Maybe<void> DataLoadOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  return InferBlobDescsWithBatchSize(GetBlobDesc4BnInOp, op_conf().data_load_conf().batch_size());
}

Maybe<void> DataLoadOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t batch_size = op_conf().data_load_conf().batch_size();
  size_t device_batch_size = GetDeviceBatchSize(batch_size, parallel_ctx);
  return InferBlobDescsWithBatchSize(GetBlobDesc4BnInOp, device_batch_size);
}

Maybe<void> DataLoadOp::InferBlobDescsWithBatchSize(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, size_t batch_size) const {
  FOR_RANGE(size_t, i, 0, output_bns().size()) {
    BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(output_bns().Get(i));
    const BlobConf& blob_conf = op_conf().data_load_conf().blobs(i);
    DimVector dim_vec(1 + blob_conf.shape().dim_size());
    dim_vec[0] = batch_size;
    FOR_RANGE(size_t, j, 1, dim_vec.size()) { dim_vec[j] = blob_conf.shape().dim(j - 1); }
    out_blob_desc->mut_shape() = Shape(dim_vec);
    out_blob_desc->set_data_type(blob_conf.data_type());
    out_blob_desc->set_is_dynamic(blob_conf.is_dynamic()
                                  || blob_conf.variable_length_axes_size() > 1);
    if (blob_conf.variable_length_axes_size() > 0) {
      out_blob_desc->SetLoD(blob_conf.variable_length_axes_size() + 1);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DataLoadOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& output_bn : output_bns()) { BatchAxis4BnInOp(output_bn)->set_value(0); }
  return Maybe<void>::Ok();
}

Maybe<void> DataLoadOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kDataLoadConf, DataLoadOp);

}  // namespace oneflow
