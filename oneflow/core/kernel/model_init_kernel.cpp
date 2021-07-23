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

class ModelInitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelInitKernel);
  ModelInitKernel() = default;
  ~ModelInitKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardDataContent(ctx, BnInOp2Blob);
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelInitOpConf& conf = this->op_conf().model_init_conf();
    const int64_t num_var = conf.out_size();
    HashMap<std::string, std::unique_ptr<SnapshotReader>> path2snapshot_reader;
    const auto GetSnapshotReader = [&](const std::string& path) -> SnapshotReader* {
      auto it = path2snapshot_reader.find(path);
      if (it != path2snapshot_reader.end()) {
        return it->second.get();
      } else {
        SnapshotReader* snapshot = new SnapshotReader(path);
        path2snapshot_reader[path].reset(snapshot);
        return snapshot;
      }
    };
    const auto InitializeWithSnapshot = [&](const std::string& snapshot_path,
                                            const std::string& key, Blob* blob) {
      SnapshotReader* reader = GetSnapshotReader(snapshot_path);
      reader->Read(key, blob);
    };
    FOR_RANGE(int64_t, i, 0, num_var) {
      Blob* out_i = BnInOp2Blob(GenRepeatedBn("out", i));
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      std::mt19937 random_seed_gen(original_variable_conf.random_seed());
      const std::string& var_lbn =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      if (original_variable_conf.has_initializer()) {
        InitializeWithConfUtil::SwitchInitializeWithConf(SwitchCase(out_i->data_type()),
                                                         original_variable_conf.initializer(),
                                                         random_seed_gen(), out_i);
      } else if (original_variable_conf.has_initialize_with_snapshot()) {
        const std::string key = original_variable_conf.initialize_with_snapshot().has_key()
                                    ? original_variable_conf.initialize_with_snapshot().key()
                                    : var_lbn;
        InitializeWithSnapshot(original_variable_conf.initialize_with_snapshot().path(), key,
                               out_i);
      } else {
        UNIMPLEMENTED();
      }
    }
  }
};

REGISTER_KERNEL(OperatorConf::kModelInitConf, ModelInitKernel);

}  // namespace oneflow
