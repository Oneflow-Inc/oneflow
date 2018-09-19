#ifndef ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
#define ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_

#include <nccl.h>
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

inline ncclDataType_t GetNcclDataType(const DataType& dt) {
  switch (dt) {
#define NCCL_DATA_TYPE_CASE(dtype) \
  case DataType::k##dtype: return ncclDataType_t::nccl##dtype
    NCCL_DATA_TYPE_CASE(Char);
    NCCL_DATA_TYPE_CASE(Float);
    NCCL_DATA_TYPE_CASE(Double);
    NCCL_DATA_TYPE_CASE(Int8);
    NCCL_DATA_TYPE_CASE(Int32);
    default: UNIMPLEMENTED();
  }
}

void NcclCheck(ncclResult_t error);

class NcclUtil final {
 public:
  using NcclReduceMthd = void(DeviceCtx*, Blob*, Blob*);
  static void AllReduce(DeviceCtx* ctx, Blob* send_blob, Blob* recv_blob) {
    auto elem_cnt = (size_t)send_blob->shape().elem_cnt();
    NcclCheck(ncclAllReduce(send_blob->dptr(), recv_blob->mut_dptr(), elem_cnt,
                            GetNcclDataType(send_blob->data_type()), ncclSum, ctx->nccl_handle(),
                            ctx->cuda_stream()));
  }

  static void ReduceScatter(DeviceCtx* ctx, Blob* send_blob, Blob* recv_blob) {
    auto elem_cnt = (size_t)send_blob->shape().elem_cnt();
    NcclCheck(ncclReduceScatter(send_blob->dptr(), recv_blob->mut_dptr(), elem_cnt,
                                GetNcclDataType(send_blob->data_type()), ncclSum,
                                ctx->nccl_handle(), ctx->cuda_stream()));
  }

  static void AllGather(DeviceCtx* ctx, Blob* send_blob, Blob* recv_blob) {
    auto elem_cnt = (size_t)send_blob->shape().elem_cnt();
    NcclCheck(ncclAllGather(send_blob->dptr(), recv_blob->mut_dptr(), elem_cnt,
                            GetNcclDataType(send_blob->data_type()), ctx->nccl_handle(),
                            ctx->cuda_stream()));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NCCL_UTIL_H_
