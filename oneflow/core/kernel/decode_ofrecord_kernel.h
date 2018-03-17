#ifndef ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/raw_ofrecord_decoder.h"

namespace std {

template<>
struct hash<oneflow::DataType> {
  typedef oneflow::DataType argument_type;
  typedef size_t result_type;

  result_type operator()(const argument_type& x) const {
    using type = typename std::underlying_type<argument_type>::type;
    return std::hash<type>()(static_cast<type>(x));
  }
};

}  // namespace std

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
};

static OFRecordDecoderIf* GetOFRecordDecoder(EncodeType encode_type,
                                             DataType data_type) {
  if (encode_type == EncodeType::kRaw) {
    static HashMap<DataType, OFRecordDecoderIf*>
        data_type_to_raw_ofrecord_decoder = {
            {DataType::kInt8, new RawOFRecordDecoder<DataType::kInt8>()},
            {DataType::kFloat, new RawOFRecordDecoder<DataType::kFloat>()},
            {DataType::kDouble, new RawOFRecordDecoder<DataType::kDouble>()},
            {DataType::kInt32, new RawOFRecordDecoder<DataType::kInt32>()}};
  } else if (encode_type == EncodeType::kJpeg) {
    // TODO
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DECODE_OFRECORD_KERNEL_H_
