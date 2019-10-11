#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class Dim0DynamicToFixedCpuKernel : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Dim0DynamicToFixedCpuKernel);
  Dim0DynamicToFixedCpuKernel() = default;
  ~Dim0DynamicToFixedCpuKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    // do nothing cuz we want remove dynamic dense_shape
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const auto& input_bns = this->op_attribute().input_bns();
    const auto& output_bns = this->op_attribute().output_bns();
    CHECK_EQ(input_bns.size() + 1, output_bns.size());
    int64_t dense_shape_dim0_val = -1;
    FOR_RANGE(int, i, 0, input_bns.size()) {
      const Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
      CHECK(in_blob->mem_case().has_host_mem());
      if (dense_shape_dim0_val == -1) {
        dense_shape_dim0_val = in_blob->shape().At(0);
      } else {
        CHECK_EQ(dense_shape_dim0_val, in_blob->shape().At(0));
      }

      Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
      CHECK(out_blob->mem_case().has_host_mem());
      CHECK(out_blob->blob_desc().is_dynamic() == false);
      CHECK_EQ(out_blob->blob_desc().Capacity(), in_blob->blob_desc().Capacity());
      out_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(input_bns.Get(i)));
    }

    Blob* mask_blob = BnInOp2Blob("mask");
    CHECK(mask_blob->mem_case().has_host_mem());
    CHECK(mask_blob->blob_desc().is_dynamic() == false);
    CHECK_EQ(1, mask_blob->shape().NumAxes());
    CHECK_EQ(DataType::kInt32, mask_blob->data_type());

    std::vector<int32_t> mask_vec(mask_blob->shape().Count(0));
    std::fill(mask_vec.begin(), mask_vec.end(), 0);
    CHECK_GT(dense_shape_dim0_val, 0);
    std::fill(mask_vec.begin(), mask_vec.begin() + dense_shape_dim0_val, 1);
    AutoMemcpy(ctx.device_ctx, mask_blob->mut_dptr(), static_cast<const void*>(mask_vec.data()),
               sizeof(int32_t) * mask_vec.size(), MakeHostMemCase(), mask_blob->mem_case());
  }
};

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kDim0DynamicToFixedConf, DeviceType::kCPU,
                            Dim0DynamicToFixedCpuKernel);

}  // namespace oneflow
