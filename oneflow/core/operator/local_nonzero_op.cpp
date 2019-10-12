#include "oneflow/core/operator/operator.h"
#include "oneflow/core/kernel/nonzero_kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
Maybe<void> InferCubSelectFlaggedTmpSize(size_t* tmp_bytes, int32_t input_size) {
  cudaError_t err =
      CubSelectFlagged<T, int32_t*>(0, input_size, nullptr, *tmp_bytes, nullptr, nullptr, nullptr);
  OF_CHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  return Maybe<void>::Ok();
}

}  // namespace

class LocalNonzeroOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalNonzeroOp);
  LocalNonzeroOp() = default;
  ~LocalNonzeroOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_nonzero_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
    EnrollOutputBn("num_nonzero", false);
    EnrollTmpBn("out_tmp");
  }

  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().local_nonzero_conf();
  }

  Maybe<void> InferOutBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext*, const SbpSignature*,
                                std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int32_t elem_cnt = in->shape().elem_cnt();
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape() = Shape({elem_cnt, in->shape().NumAxes()});
    out->set_data_type(DataType::kInt32);
    // nnz
    BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
    num_nonzero->mut_shape() = Shape({1});
    num_nonzero->set_data_type(DataType::kInt32);
    return Maybe<void>::Ok();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int32_t elem_cnt = in->shape().elem_cnt();
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    out->mut_shape() = Shape({elem_cnt, in->shape().NumAxes()});
    out->set_data_type(DataType::kInt32);
    // nnz
    BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
    num_nonzero->mut_shape() = Shape({1});
    num_nonzero->set_data_type(DataType::kInt32);

    {
      size_t out_tmp_bytes = 0;
      switch (in->data_type()) {
        case DataType::kFloat:
          InferCubSelectFlaggedTmpSize<float>(&out_tmp_bytes, in->shape().elem_cnt());
          break;
        case DataType::kInt8:
          InferCubSelectFlaggedTmpSize<int8_t>(&out_tmp_bytes, in->shape().elem_cnt());
          break;
        case DataType::kInt32:
          InferCubSelectFlaggedTmpSize<int32_t>(&out_tmp_bytes, in->shape().elem_cnt());
          break;
        default:
          OF_UNIMPLEMENTED() << "Nonzero Op/Kernel do not support "
                             << static_cast<int>(in->data_type());
      }

      OF_CHECK_GT(out_tmp_bytes, 0) << "out_tmp_bytes should be greater than zero.";
      BlobDesc* out_tmp = GetBlobDesc4BnInOp("out_tmp");
      out_tmp->mut_shape() = Shape(std::vector<int64_t>{static_cast<int64_t>(out_tmp_bytes)});
      out_tmp->set_data_type(DataType::kChar);
    }
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
    kernel_conf->set_data_type(in_blob_desc->data_type());
    kernel_conf->mutable_nonzero_gpu_kernel_conf()->set_num_axes(in_blob_desc->shape().NumAxes());
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    for (const std::string& obn : output_bns()) { BatchAxis4BnInOp(obn)->set_value(0); }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalNonzeroConf, LocalNonzeroOp);

}  // namespace oneflow
