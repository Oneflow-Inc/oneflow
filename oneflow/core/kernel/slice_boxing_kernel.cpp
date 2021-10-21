/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/slice_boxing_kernel_util.h"

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
#ifdef WITH_CUDA
          || (device_type == DeviceType::kGPU && in_i->mem_case().has_host_mem()
              && in_i->mem_case().host_mem().has_cuda_pinned_mem())
          || (device_type == DeviceType::kGPU && in_i->mem_case().has_device_cuda_mem()
              && out->mem_case().has_device_cuda_mem()
              && out->mem_case().device_cuda_mem().device_id()
                     == in_i->mem_case().device_cuda_mem().device_id());
#else
          ;
#endif
      if (in_i->shape() == out->shape() && can_direct_access) {
        SliceBoxingKernelUtil<device_type, T>::Add(ctx.device_ctx, out->shape().elem_cnt(),
                                                   in_i->dptr<T>(), out->dptr<T>(),
                                                   out->mut_dptr<T>());
      } else {
        Blob* buf = BnInOp2Blob("buf");
        this->tensor_slice_copier_vec().at(i)->Copy(ctx.device_ctx, *this->memory_copier(), buf,
                                                    in_i);
        SliceBoxingKernelUtil<device_type, T>::Add(ctx.device_ctx, out->shape().elem_cnt(),
                                                   buf->dptr<T>(), out->dptr<T>(),
                                                   out->mut_dptr<T>());
      }
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingCopyConf, SliceBoxingCopyKernel,
                           POD_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSliceBoxingAddConf, SliceBoxingAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
