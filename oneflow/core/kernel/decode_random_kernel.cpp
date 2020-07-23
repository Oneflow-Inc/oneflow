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
#include "oneflow/core/kernel/decode_random_kernel.h"

namespace oneflow {

namespace {

void RandomFillBlob(DeviceCtx* ctx, DeviceType device_type, const InitializerConf& initializer_conf,
                    uint32_t random_seed, Blob* blob) {
  static const HashMap<std::string,
                       void (*)(DeviceCtx * ctx, const InitializerConf& initializer_conf,
                                uint32_t random_seed, Blob* blob)>
      fill_funcs = {
#define RANDOM_FILL_ENTRY(type_dev, data_type_pair)         \
  {GetHashKey(type_dev, OF_PP_PAIR_SECOND(data_type_pair)), \
   &KernelUtil<type_dev, OF_PP_PAIR_FIRST(data_type_pair)>::InitializeWithConf},
          OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(RANDOM_FILL_ENTRY, DEVICE_TYPE_SEQ,
                                           ARITHMETIC_DATA_TYPE_SEQ)};
  fill_funcs.at(GetHashKey(device_type, blob->data_type()))(ctx, initializer_conf, random_seed,
                                                            blob);
}

}  // namespace

template<DeviceType device_type>
void DecodeRandomKernel<device_type>::VirtualKernelInit() {
  gen_.reset(new std::mt19937(this->kernel_conf().decode_random_conf().random_seed()));
  dis_.reset(new std::uniform_int_distribution<uint32_t>());
  is_init_ = false;
}

template<DeviceType device_type>
uint32_t DecodeRandomKernel<device_type>::GenNextRandomSeed() const {
  return (*dis_)(*gen_);
}

template<DeviceType device_type>
void DecodeRandomKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const DecodeRandomOpConf& conf = this->op_conf().decode_random_conf();
  if (is_init_ == false) {
    RandomFillBlob(ctx.device_ctx, device_type, conf.data_initializer(), this->GenNextRandomSeed(),
                   BnInOp2Blob("out"));
    is_init_ = true;
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDecodeRandomConf, DecodeRandomKernel);

}  // namespace oneflow
