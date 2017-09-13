#include "oneflow/core/kernel/record_kernel.h"

namespace oneflow {

namespace {

template<typename T>
void RecordBlobImpl(PersistentOutStream& out_stream, const Blob* blob) {
  CHECK_EQ(GetDataType<T>::val, blob->data_type());
  blob->shape().SerializeWithTextFormat(out_stream);
  out_stream << '\n';
  const T* dptr = blob->dptr<T>();
  for (int64_t i = 0; i < blob->shape().At(0); ++i) {
    for (int64_t j = 0; j < blob->shape().Count(1); ++j) {
      out_stream << std::to_string(*dptr++) << ' ';
    }
    out_stream << '\n';
  }
}

void RecordBlob(PersistentOutStream& out_stream, const Blob* blob) {
  static const HashMap<int, void (*)(PersistentOutStream&, const Blob*)>
      record_funcs = {
#define RECORD_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, &RecordBlobImpl<type_cpp>},
          OF_PP_FOR_EACH_TUPLE(RECORD_KERNEL_ENTRY, ALL_DATA_TYPE_SEQ)};
  record_funcs.at(blob->data_type())(out_stream, blob);
}

}  // namespace

void RecordKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t parallel_id = reinterpret_cast<int64_t>(kernel_ctx.other);
  const std::string& root_path = op()->op_conf().record_conf().record_path();
  OF_ONCE_GUARD(root_path, GlobalFS()->CreateDirIfNotExist(root_path));
  for (const std::string& ibn : op()->input_bns()) {
    const std::string& lbn = op()->Lbn4BnInOp(ibn);
    const Blob* blob = BnInOp2Blob(ibn);
    kernel_ctx.device_ctx->cpu_stream()->SendWork(
        [lbn, blob, parallel_id, root_path]() {
          std::pair<std::string, std::string> parsed_lbn = ParseLbn(lbn);
          const std::string& op_name = parsed_lbn.first;
          const std::string& bn_in_op = parsed_lbn.second;
          std::string op_dir = JoinPath(root_path, op_name);
          OF_ONCE_GUARD(op_dir, GlobalFS()->CreateDir(op_dir));
          std::string bn_in_op_dir = JoinPath(op_dir, bn_in_op);
          OF_ONCE_GUARD(bn_in_op_dir, GlobalFS()->CreateDir(bn_in_op_dir));
          std::string file_path =
              JoinPath(bn_in_op_dir, "part_" + std::to_string(parallel_id));
          PersistentOutStream out_stream(GlobalFS(), file_path);
          RecordBlob(out_stream, blob);
        });
  }
}

COMMAND(AddKernelCreator(OperatorConf::kRecordConf,
                         []() { return new RecordKernel; }));

}  // namespace oneflow
