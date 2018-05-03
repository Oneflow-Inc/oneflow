#include "oneflow/core/actor/copy_hd_actor.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdActor::VirtualActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&CopyHdActor::HandlerNormal);
}

void CopyHdActor::InitDeviceCtx(const ThreadCtx& thread_ctx) {
  CHECK_EQ(exec_kernel_vec().size(), 1);
  const OperatorConf& op_conf = exec_kernel_vec().begin()->kernel->op_conf();
  CHECK(op_conf.has_copy_hd_conf());
  const cudaStream_t* cuda_stream = nullptr;
  int64_t work_stream_id = -1;
  if (op_conf.copy_hd_conf().type() == CopyHdOpConf::H2D) {
    cuda_stream = thread_ctx.copy_h2d_cuda_stream;
    work_stream_id = GetReservedWorkStreamId(0);
  } else if (op_conf.copy_hd_conf().type() == CopyHdOpConf::D2H) {
    cuda_stream = thread_ctx.copy_d2h_cuda_stream;
    work_stream_id = GetReservedWorkStreamId(1);
  } else {
    UNIMPLEMENTED();
  }
  CHECK_NOTNULL(cuda_stream);
  mut_device_ctx().reset(new CudaDeviceCtx(work_stream_id, cuda_stream));
}

void CopyHdActor::Act() {
  Regst* in_regst = GetNaiveSoleCurReadable();
  AsyncLaunchKernel(GenDefaultKernelCtx());
  AsyncSendRegstMsgToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(in_regst->piece_id());
    out_regst->set_model_version_id(in_regst->model_version_id());
    return true;
  });
}

REGISTER_ACTOR(TaskType::kCopyHd, CopyHdActor);

#endif

}  // namespace oneflow
