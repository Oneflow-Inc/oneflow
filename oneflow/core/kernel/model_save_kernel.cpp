#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

std::string GenNewSnapshotName() {
  const auto now_clock = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now_clock);
  char datetime[sizeof("2006_01_02_15_04_05")];
  std::strftime(datetime, sizeof(datetime), "%Y_%m_%d_%H_%M_%S", std::localtime(&now_time));
  std::ostringstream oss;
  oss << "snapshot_" << datetime << "_" << std::setw(3) << std::setfill('0')
      << std::chrono::duration_cast<std::chrono::milliseconds>(now_clock.time_since_epoch()).count()
             % 1000;
  return oss.str();
}

}  // namespace

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
  const Blob* path_blob = BnInOp2Blob("path");
  const std::string path(path_blob->dptr<char>(), path_blob->shape().elem_cnt());
  SnapshotWriter writer(path);
  FOR_RANGE(int64_t, i, 0, op_attribute().input_bns().size()) {
    const Blob* in_i = BnInOp2Blob(GenRepeatedBn("in", i));
    writer.Write(op_conf().model_save_conf().key(i), in_i);
  }
  writer.Close();
}

REGISTER_KERNEL(OperatorConf::kModelSaveConf, ModelSaveKernel);

}  // namespace oneflow
