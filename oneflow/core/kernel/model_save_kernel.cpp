#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelSaveKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveKernel);
  ModelSaveKernel() = default;
  ~ModelSaveKernel() override = default;

 private:
  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

void ModelSaveKernel::Forward(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const ModelSaveOpConf& conf = this->op_conf().model_save_conf();
  const Blob* path_blob = BnInOp2Blob("path");
  const std::string path(path_blob->dptr<char>(), path_blob->shape_view().elem_cnt());
  SnapshotWriter writer(path);
  FOR_RANGE(int64_t, i, 0, conf.in_size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    writer.Write(conf.key(i), in_i);
  }
  writer.Close();
}

REGISTER_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
