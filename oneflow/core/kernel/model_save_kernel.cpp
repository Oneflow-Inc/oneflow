#include "oneflow/core/kernel/model_save_kernel.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

template<typename T>
void ModelSaveKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto save_ctx =
      static_cast<std::tuple<Snapshot*, int64_t, int64_t, ParallelPolicy>*>(
          kernel_ctx.other);
  Snapshot* snapshot = std::get<0>(*save_ctx);
  int64_t parallel_id = std::get<1>(*save_ctx);
  int64_t parallel_num = std::get<2>(*save_ctx);
  ParallelPolicy policy = std::get<3>(*save_ctx);
  int32_t part_id = -1;
  int32_t total_part_num = -1;
  if (policy == kDataParallel) {
    part_id = 0;
    total_part_num = 1;
    CHECK_EQ(parallel_id, 0);
  } else if (policy == kModelParallel) {
    part_id = parallel_id;
    total_part_num = parallel_num;
  } else {
    UNEXPECTED_RUN();
  }
  for (const std::string& ibn : op()->input_bns()) {
    const std::string& lbn = op()->Lbn4BnInOp(ibn);
    Blob* blob_ptr = BnInOp2Blob(ibn);
    kernel_ctx.device_ctx->cpu_stream()->SendWork([=]() {
      {
        std::unique_ptr<PersistentOutStream> out_stream =
            snapshot->GetOutStream(lbn, part_id);
        out_stream->Write(blob_ptr->dptr<char>(),
                          blob_ptr->shape().elem_cnt() * sizeof(T));
      }
      snapshot->OnePartDone(lbn, part_id, total_part_num);
    });
  }
}

Kernel* CreateModelSaveKernel() {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define MODEL_SAVE_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new ModelSaveKernel<type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(MODEL_SAVE_KERNEL_ENTRY, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(JobDesc::Singleton()->default_data_type())();
}

COMMAND(AddKernelCreator(OperatorConf::kModelSaveConf, CreateModelSaveKernel));

}  // namespace oneflow
