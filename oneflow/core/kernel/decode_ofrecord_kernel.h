#ifndef ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

struct DecodeStatus {
  Regst* in_regst_;
  int32_t cur_col_id_;
  int32_t max_col_id_;
};

template<typename T>
class DecodeOFRecordKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOFRecordKernel);
  DecodeOFRecordKernel() = default;
  ~DecodeOFRecordKernel() = default;

 private:
  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

  void ReadColNumToOutBlob(Blob* out_blob, const std::string& name, EncodeType,
                           RecordBlob<OFRecord>*, int32_t* max_clo_id);
  void ReadDataIdToOutBlob(Blob* out_blob, RecordBlob<OFRecord>*, DeviceCtx*);
  void ReadDataContentToOutBlob(Blob* out_blob, const std::string& name,
                                EncodeType, RecordBlob<OFRecord>*,
                                int32_t col_id, DeviceCtx*);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
