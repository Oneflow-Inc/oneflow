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

namespace oneflow {

template<DeviceType device_type>
class DecodeRandomKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomKernel);
  DecodeRandomKernel() : is_init_(false){};
  ~DecodeRandomKernel() = default;

  void Forward(KernelContext* ctx) const override { ForwardDataContent(ctx); }

  void ForwardDataContent(KernelContext* ctx) const override;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  uint32_t GenNextRandomSeed() const;

  std::unique_ptr<std::mt19937> gen_;
  std::unique_ptr<std::uniform_int_distribution<uint32_t>> dis_;

  mutable bool is_init_;
};

namespace {

void RandomFillBlob(ep::Stream* stream, DeviceType device_type,
                    const InitializerConf& initializer_conf, uint32_t random_seed, Blob* blob) {
  static const HashMap<std::string,
                       void (*)(ep::Stream * stream, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob)>
      fill_funcs = {
#define RANDOM_FILL_ENTRY(type_dev, data_type_pair)         \
  {GetHashKey(type_dev, OF_PP_PAIR_SECOND(data_type_pair)), \
   &KernelUtil<type_dev, OF_PP_PAIR_FIRST(data_type_pair)>::InitializeWithConf},
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(RANDOM_FILL_ENTRY, DEVICE_TYPE_SEQ,
                                           ARITHMETIC_DATA_TYPE_SEQ)};
  fill_funcs.at(GetHashKey(device_type, blob->data_type()))(stream, initializer_conf, random_seed,
                                                            blob);
}

}  // namespace

template<DeviceType device_type>
void DecodeRandomKernel<device_type>::VirtualKernelInit(KernelContext* ctx) {
  gen_.reset(new std::mt19937(this->kernel_conf().decode_random_conf().random_seed()));
  dis_.reset(new std::uniform_int_distribution<uint32_t>());
  is_init_ = false;
}

template<DeviceType device_type>
uint32_t DecodeRandomKernel<device_type>::GenNextRandomSeed() const {
  return (*dis_)(*gen_);
}

template<DeviceType device_type>
void DecodeRandomKernel<device_type>::ForwardDataContent(KernelContext* ctx) const {
  const DecodeRandomOpConf& conf = this->op_conf().decode_random_conf();
  if (is_init_ == false) {
    RandomFillBlob(ctx->stream(), device_type, conf.data_initializer(), this->GenNextRandomSeed(),
                   ctx->BnInOp2Blob("out"));
    is_init_ = true;
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDecodeRandomConf, DecodeRandomKernel);

}  // namespace oneflow
