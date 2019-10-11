#include "oneflow/core/operator/operator.h"
#include "oneflow/core/kernel/nonzero_kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
Maybe<void> InferCubReduceCountTmpSize(size_t* tmp_bytes, int32_t input_size) {
  cudaError_t err = CubReduceCount<T>(nullptr, *tmp_bytes, nullptr, nullptr, input_size, 0);
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
    if (this->device_type() == DeviceType::kGPU) { EnrollTmpBn("shape"); }
    EnrollOutputBn("out", false);
    EnrollOutputBn("num_nonzero", false);
    EnrollTmpBn("nnz_tmp");
    EnrollTmpBn("out_tmp");
  }

  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().local_nonzero_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext*) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const int64_t elem_cnt = in->shape().elem_cnt();
    if (this->device_type() == DeviceType::kGPU) {
      // data tmp: shape
      BlobDesc* shape = GetBlobDesc4BnInOp("shape");
      shape->mut_shape() = Shape({in->shape().NumAxes()});
      shape->set_data_type(DataType::kInt64);
    }
    // output
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    out->mut_shape() = Shape({elem_cnt, in->shape().NumAxes()});
    out->set_data_type(DataType::kInt32);
    // output: num_nonzero
    BlobDesc* num_nonzero = GetBlobDesc4BnInOp("num_nonzero");
    num_nonzero->mut_shape() = Shape({1});
    num_nonzero->set_data_type(DataType::kInt64);

    {
      size_t tmp_bytes = 0;
      switch (in->data_type()) {
        case DataType::kFloat:
          InferCubReduceCountTmpSize<float>(&tmp_bytes, in->shape().elem_cnt());
          break;
        case DataType::kDouble:
          InferCubReduceCountTmpSize<double>(&tmp_bytes, in->shape().elem_cnt());
          break;
        case DataType::kInt32:
          InferCubReduceCountTmpSize<int32_t>(&tmp_bytes, in->shape().elem_cnt());
          break;
        case DataType::kInt64:
          InferCubReduceCountTmpSize<int64_t>(&tmp_bytes, in->shape().elem_cnt());
          break;
        default:
          OF_UNIMPLEMENTED() << "Nonzero Op/Kernel do not support "
                             << static_cast<int>(in->data_type());
          OF_CHECK_GT(tmp_bytes, 0) << "tmp_bytes should be greater than zero.";
          BlobDesc* nnz_tmp = GetBlobDesc4BnInOp("nnz_tmp");
          nnz_tmp->mut_shape() = Shape(std::vector<int64_t>{static_cast<int64_t>(tmp_bytes)});
          nnz_tmp->set_data_type(DataType::kChar);
      }
    }

    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf* kernel_conf) const override {
    kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
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
