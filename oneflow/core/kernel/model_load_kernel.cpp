#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelLoadKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadKernel);
  ModelLoadKernel() = default;
  ~ModelLoadKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelLoadOpConf& conf = this->op_conf().model_load_conf();
    const Blob* path_blob = BnInOp2Blob("path");
    const std::string path(path_blob->dptr<char>(), path_blob->dense_shape_view().elem_cnt());
    SnapshotReader reader(path);
    FOR_RANGE(int64_t, i, 0, conf.out_size()) {
      LOG(INFO) << "load " << conf.variable_op_name(i);
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      Blob* out_i = BnInOp2Blob(GenRepeatedBn("out", i));
      const std::string key =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      reader.Read(key, out_i);
    }
  }
};

REGISTER_KERNEL(OperatorConf::kModelLoadConf, ModelLoadKernel);

}  // namespace oneflow
