#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ModelLoadKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadKernel);
  ModelLoadKernel() = default;
  ~ModelLoadKernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelLoadOpConf& conf = this->op_conf().model_load_conf();
    const Blob* path_blob = BnInOp2Blob("path");
    const size_t path_len =
        strnlen(path_blob->dptr<char>(), path_blob->ByteSizeOfDataContentField());
    const std::string path(path_blob->dptr<char>(), path_len);
    SnapshotReader reader(path);
    FOR_RANGE(int64_t, i, 0, conf.out_size()) {
      const VariableOpConf& original_variable_conf = conf.original_variable_conf(i);
      Blob* out_i = BnInOp2Blob(GenRepeatedBn("out", i));
      const std::string key =
          GenLogicalBlobName(conf.variable_op_name(i), original_variable_conf.out());
      reader.Read(key, out_i);
    }
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kModelLoadConf, ModelLoadKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
