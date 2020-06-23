#include "oneflow/core/kernel/print_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

void PrintKernel::VirtualKernelInit() {
  const auto& conf = op_conf().print_conf();
  const std::string& root_path = conf.print_dir();
  OfCallOnce(root_path, SnapshotFS(), &fs::FileSystem::RecursivelyCreateDir);
  int32_t part_name_suffix_length = conf.part_name_suffix_length();
  std::string num = "0";  // TODO() useless kernel
  int32_t zero_count = std::max(part_name_suffix_length - static_cast<int32_t>(num.length()), 0);
  std::string file_path =
      JoinPath(root_path, conf.part_name_prefix() + std::string(zero_count, '0') + num);
  out_stream_.reset(new PersistentOutStream(SnapshotFS(), file_path));
}

void PrintKernel::ForwardDataContent(const KernelCtx& ctx,
                                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

REGISTER_KERNEL(OperatorConf::kPrintConf, PrintKernel);

}  // namespace oneflow
