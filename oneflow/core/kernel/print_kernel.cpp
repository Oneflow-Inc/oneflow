#include "oneflow/core/kernel/print_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<typename T>
void PrintBlob(PersistentOutStream& out_stream, const Blob* blob,
               const std::string& blob_name) {
  CHECK_EQ(GetDataType<T>::value, blob->data_type());
  const T* dptr = blob->dptr<T>();
  out_stream << "\n" << blob_name << " {\n";
  for (int64_t i = 0; i < blob->shape().At(0); ++i) {
    if (blob->has_data_id_field()) {
      out_stream << "data_id: ";
      size_t data_id_size = 0;
      for (; data_id_size != Global<JobDesc>::Get()->SizeOfOneDataId();
           ++data_id_size) {
        if (*(blob->data_id(i) + data_id_size) == '\0') { break; }
      }
      if (data_id_size == 0) { continue; }
      out_stream.Write(blob->data_id(i), data_id_size);
      out_stream.Write("\n", 1);
    }
    out_stream << "data_content: ";
    for (int64_t j = 0; j < blob->shape().Count(1); ++j) {
      out_stream << std::to_string(*dptr++) << ',';
    }
    out_stream << '\n';
  }
  out_stream << "}\n\n";
}

}  // namespace

void PrintKernel::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const std::string& root_path = op_conf().print_conf().print_path();
  OF_CALL_ONCE(root_path, GlobalFS()->MakeEmptyDir(root_path));
  FOR_RANGE(size_t, i, 0, op_conf().print_conf().lbn().size()) {
    const std::string& lbn = op_conf().print_conf().lbn(i);
    std::pair<std::string, std::string> parsed_lbn = ParseLbn(lbn);
    const std::string& op_name = parsed_lbn.first;
    const std::string& bn_in_op = parsed_lbn.second;
    std::string op_dir = JoinPath(root_path, op_name);
    OF_CALL_ONCE(op_dir, GlobalFS()->CreateDir(op_dir));
    std::string bn_in_op_dir = JoinPath(op_dir, bn_in_op);
    OF_CALL_ONCE(bn_in_op_dir, GlobalFS()->CreateDir(bn_in_op_dir));
    std::string file_path = JoinPath(
        bn_in_op_dir, "part-" + std::to_string(parallel_ctx->parallel_id()));
    out_streams_.emplace_back(new PersistentOutStream(GlobalFS(), file_path));
  }
}

#define MAKE_PRINTBLOB_SWITCH_ENTRY(func_name, T) func_name<T>
DEFINE_STATIC_SWITCH_FUNC(void, PrintBlob, MAKE_PRINTBLOB_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(ALL_DATA_TYPE_SEQ));
#undef MAKE_PRINTBLOB_SWITCH_ENTRY

void PrintKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, kernel_conf().input_bns().size()) {
    const std::string& ibn = kernel_conf().input_bns(i);
    const Blob* blob = BnInOp2Blob(ibn);
    SwitchPrintBlob(SwitchCase(blob->data_type()), *out_streams_[i], blob, ibn);
    out_streams_[i]->Flush();
  }
}

COMMAND(AddKernelCreator(OperatorConf::kPrintConf,
                         []() { return new PrintKernel; }));

}  // namespace oneflow
