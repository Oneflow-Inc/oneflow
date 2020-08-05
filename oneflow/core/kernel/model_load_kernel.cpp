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
#include "oneflow/core/common/str_util.h"
#include <iostream>

namespace oneflow {

namespace {

template<typename T>
void InitializeWithConf(const InitializerConf& conf, const uint32_t random_seed, Blob* blob) {
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, conf, random_seed, blob);
}

struct InitializeWithConfUtil final {
#define MAKE_INITIALIZE_SWITCH_ENTRY(func_name, T) func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(void, InitializeWithConf, MAKE_INITIALIZE_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_INITIALIZE_SWITCH_ENTRY
};

}  // namespace

class ModelLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadKernel);
  ModelLoadKernel() = default;
  ~ModelLoadKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelLoadOpConf& conf = this->op_conf().model_load_conf();
    const Blob* path_blob = BnInOp2Blob("path");
    const std::string path(path_blob->dptr<char>(), path_blob->shape_view().elem_cnt());
    SnapshotReader reader(path);
    FOR_RANGE(int64_t, i, 0, conf.out_size()) {
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      Blob* out_i = BnInOp2Blob(GenRepeatedBn("out", i));
      const std::string key =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      if (reader.HasKey(key)) {
        reader.Read(key, out_i);
      } else {
        std::cout << "WARNING! CANNOT find variable path in : " << JoinPath(path, key)
                  << ". It will be initialized. \n";
        std::mt19937 random_seed_gen(original_variable_conf.random_seed());
        CHECK(original_variable_conf.has_initializer())
            << "ERROR! variable must has initializer when load failed.";
        InitializeWithConfUtil::SwitchInitializeWithConf(SwitchCase(out_i->data_type()),
                                                         original_variable_conf.initializer(),
                                                         random_seed_gen(), out_i);
      }
    }
  }
};

REGISTER_KERNEL(OperatorConf::kModelLoadConf, ModelLoadKernel);

}  // namespace oneflow
