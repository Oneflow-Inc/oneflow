#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ModelInitKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelInitKernel);
  ModelInitKernel() = default;
  ~ModelInitKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
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
    const auto InitializeWithConf = [&](const InitializerConf& conf, const uint32_t random_seed,
                                        Blob* blob) {
      if (blob->data_type() == DataType::kFloat) {
        KernelUtil<device_type, float>::InitializeWithConf(ctx.device_ctx, conf, random_seed, blob);
      } else if (blob->data_type() == DataType::kInt32) {
        KernelUtil<device_type, int32_t>::InitializeWithConf(ctx.device_ctx, conf, random_seed,
                                                             blob);
      } else if (blob->data_type() == DataType::kInt64) {
        KernelUtil<device_type, int64_t>::InitializeWithConf(ctx.device_ctx, conf, random_seed,
                                                             blob);
      } else {
        UNIMPLEMENTED();
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
        InitializeWithConf(original_variable_conf.initializer(), random_seed_gen(), out_i);
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

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kModelInitConf, ModelInitKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
