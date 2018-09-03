#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

#ifdef WITH_CUDA

const cudaStream_t* CudaStreamHandle::cuda_stream() {
  if (!cuda_stream_) {
    cuda_stream_.reset(new cudaStream_t);
    CudaCheck(cudaStreamCreate(cuda_stream_.get()));
  }
  return cuda_stream_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmh_handle() {
  if (!cublas_pmh_handle_) {
    cublas_pmh_handle_.reset(new cublasHandle_t);
    CudaCheck(cublasCreate(cublas_pmh_handle_.get()));
    CudaCheck(cublasSetStream(*cublas_pmh_handle_, *cuda_stream()));
  }
  return cublas_pmh_handle_.get();
}

const cublasHandle_t* CudaStreamHandle::cublas_pmd_handle() {
  if (!cublas_pmd_handle_) {
    cublas_pmd_handle_.reset(new cublasHandle_t);
    CudaCheck(cublasCreate(cublas_pmd_handle_.get()));
    CudaCheck(cublasSetStream(*cublas_pmd_handle_, *cuda_stream()));
    CudaCheck(cublasSetPointerMode(*cublas_pmd_handle_, CUBLAS_POINTER_MODE_DEVICE));
  }
  return cublas_pmd_handle_.get();
}

const cudnnHandle_t* CudaStreamHandle::cudnn_handle() {
  if (!cudnn_handle_) {
    cudnn_handle_.reset(new cudnnHandle_t);
    CudaCheck(cudnnCreate(cudnn_handle_.get()));
    CudaCheck(cudnnSetStream(*cudnn_handle_, *cuda_stream()));
  }
  return cudnn_handle_.get();
}

const ncclComm_t* CudaStreamHandle::nccl_handle() {
  if (!nccl_handle_) {
    int32_t rank_num = Global<JobDesc>::Get()->GpuDeviceNum();
    int32_t my_rank = dev_id_;
    ncclUniqueId nccl_unique_id = Global<MachineCtx>::Get()->GetNcclUniqueId();
    nccl_handle_.reset(new ncclComm_t);
    CudaCheck(ncclCommInitRank(nccl_handle_.get(), rank_num, nccl_unique_id, my_rank));
  }
  return nccl_handle_.get();
}

const ncclComm_t* CudaStreamHandle::nccl_scatter_handle() {
  if (!nccl_scatter_handle_) {
    int32_t rank_num = Global<JobDesc>::Get()->GpuDeviceNum();
    int32_t my_rank = dev_id_;
    ncclUniqueId nccl_scatter_unique_id = Global<MachineCtx>::Get()->GetNcclScatterUniqueId();
    nccl_scatter_handle_.reset(new ncclComm_t);
    CudaCheck(
        ncclCommInitRank(nccl_scatter_handle_.get(), rank_num, nccl_scatter_unique_id, my_rank));
  }
  return nccl_scatter_handle_.get();
}

const ncclComm_t* CudaStreamHandle::nccl_gather_handle() {
  if (!nccl_gather_handle_) {
    int32_t rank_num = Global<JobDesc>::Get()->GpuDeviceNum();
    int32_t my_rank = dev_id_;
    ncclUniqueId nccl_gather_unique_id = Global<MachineCtx>::Get()->GetNcclGatherUniqueId();
    nccl_gather_handle_.reset(new ncclComm_t);
    CudaCheck(
        ncclCommInitRank(nccl_gather_handle_.get(), rank_num, nccl_gather_unique_id, my_rank));
  }
  return nccl_gather_handle_.get();
}

void CudaStreamHandle::AddCallBack(std::function<void()> callback) {
  CudaCBEvent cb_event;
  cb_event.callback = callback;
  CudaCheck(
      cudaEventCreateWithFlags(&(cb_event.event), cudaEventBlockingSync | cudaEventDisableTiming));
  CudaCheck(cudaEventRecord(cb_event.event, *cuda_stream()));
  cb_event_chan_->Send(cb_event);
}

CudaStreamHandle::~CudaStreamHandle() {
  if (cudnn_handle_) { CudaCheck(cudnnDestroy(*cudnn_handle_)); }
  if (cublas_pmh_handle_) { CudaCheck(cublasDestroy(*cublas_pmh_handle_)); }
  if (cublas_pmd_handle_) { CudaCheck(cublasDestroy(*cublas_pmd_handle_)); }
  if (cuda_stream_) { CudaCheck(cudaStreamDestroy(*cuda_stream_)); }
  if (nccl_handle_) { ncclCommDestroy(*nccl_handle_); }
  if (nccl_scatter_handle_) { ncclCommDestroy(*nccl_scatter_handle_); }
  if (nccl_gather_handle_) { ncclCommDestroy(*nccl_gather_handle_); }
}

#endif  // WITH_CUDA

}  // namespace oneflow
