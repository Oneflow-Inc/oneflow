#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
class SliceBoxingKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingKernel);
  SliceBoxingKernel() = default;
  ~SliceBoxingKernel() override = default;

 protected:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const = 0;
  MemoryCopier* memory_copier() const;
  const std::vector<std::shared_ptr<TensorSliceCopier>>& tensor_slice_copier_vec() const;

 private:
  void VirtualKernelInit() override;

  std::vector<std::shared_ptr<TensorSliceCopier>> tensor_slice_copier_vec_;
  std::unique_ptr<MemoryCopier> memory_copier_;
};

template<DeviceType device_type, typename T>
class SliceBoxingCopyKernel final : public SliceBoxingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingCopyKernel);
  SliceBoxingCopyKernel() = default;
  ~SliceBoxingCopyKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class SliceBoxingAddKernel final : public SliceBoxingKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceBoxingAddKernel);
  SliceBoxingAddKernel() = default;
  ~SliceBoxingAddKernel() override = default;

 private:
  virtual const SliceBoxingConf& GetCustomizedBoxingConf() const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void SliceBoxingKernel<device_type, T>::VirtualKernelInit() {
  memory_copier_.reset(NewDefaultMemoryCopier(device_type));
  const SliceBoxingConf& conf = GetCustomizedBoxingConf();
  const TensorSliceView out_slice(conf.out_slice());
  for (const TensorSliceViewProto& in_slice_proto : conf.in_slice()) {
    const TensorSliceView in_slice(in_slice_proto);
    tensor_slice_copier_vec_.emplace_back(
        new TensorSliceCopier(out_slice, in_slice, this->kernel_conf().data_type()));
  }
}

template<DeviceType device_type, typename T>
MemoryCopier* SliceBoxingKernel<device_type, T>::memory_copier() const {
  return memory_copier_.get();
}

template<DeviceType device_type, typename T>
const std::vector<std::shared_ptr<TensorSliceCopier>>&
SliceBoxingKernel<device_type, T>::tensor_slice_copier_vec() const {
  return tensor_slice_copier_vec_;
}

template<DeviceType device_type, typename T>
const SliceBoxingConf& SliceBoxingCopyKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_copy_conf().slice_boxing_conf();
}

template<DeviceType device_type, typename T>
void SliceBoxingCopyKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), out, in_i);
  }
}

template<DeviceType device_type, typename T>
const SliceBoxingConf& SliceBoxingAddKernel<device_type, T>::GetCustomizedBoxingConf() const {
  return this->op_conf().slice_boxing_add_conf().slice_boxing_conf();
}

template<DeviceType device_type, typename T>
void SliceBoxingAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out = BnInOp2Blob("out");
  FOR_RANGE(int64_t, i, 0, this->op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    if (i == 0) {
      this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), out,
                                                  in_i);
    } else {
      bool can_direct_access =
          (device_type == kCPU)
          || (device_type == DeviceType::kGPU && in_i->mem_case().has_host_mem()
              && in_i->mem_case().host_mem().has_cuda_pinned_mem())
          || (device_type == DeviceType::kGPU && in_i->mem_case().has_device_cuda_mem()
              && out->mem_case().has_device_cuda_mem()
              && out->mem_case().device_cuda_mem().device_id()
                     == in_i->mem_case().device_cuda_mem().device_id());
      if (in_i->shape() == out->shape() && can_direct_access) {
        KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out->shape().elem_cnt(), GetOneVal<T>(),
                                         in_i->dptr<T>(), 1, out->mut_dptr<T>(), 1);
      } else {
        Blob* buf = BnInOp2Blob("buf");
        this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), buf,
                                                    in_i);
        KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out->shape().elem_cnt(), GetOneVal<T>(),
                                         buf->dptr<T>(), 1, out->mut_dptr<T>(), 1);
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyKernel,
                           POD_DATA_TYPE_SEQ)
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
