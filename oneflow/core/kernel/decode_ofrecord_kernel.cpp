#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

namespace {

template<typename T>
int64_t GetByteSizeOf(const T& feature_list) {
  return feature_list.size() * sizeof(typename T::value_type);
}

std::pair<const void*, int64_t> GetDptrAndColNum(const Feature& feature,
                                                 EncodeType encode_type,
                                                 int64_t col_size) {
  if (encode_type == kJpeg) {
    CHECK(feature.has_bytes_list());
    return {feature.bytes_list().value().data(),
            feature.bytes_list().value().size()};
  } else if (encode_type == kRaw) {
    if (feature.has_bytes_list()) {
      CHECK_EQ(feature.bytes_list().value_size(), 1);
      return {feature.bytes_list().value(0).data(),
              GetByteSizeOf(feature.bytes_list().value(0)) / col_size};
    } else if (feature.has_float_list()) {
      return {feature.float_list().value().data(),
              GetByteSizeOf(feature.float_list().value()) / col_size};
    } else if (feature.has_double_list()) {
      return {feature.double_list().value().data(),
              GetByteSizeOf(feature.double_list().value()) / col_size};
    } else if (feature.has_int32_list()) {
      return {feature.int32_list().value().data(),
              GetByteSizeOf(feature.int32_list().value()) / col_size};
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<typename T>
void DecodeOFRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_ctx.other);
  auto status = static_cast<DecodeStatus*>(kernel_ctx.other);

  RecordBlob<OFRecord>* record_blob =
      status->in_regst_->GetRecordBlob<OFRecord>();
  const DecodeOFRecordOpConf& decode_conf = op_conf().decode_ofrecord_conf();
  CHECK_EQ(kernel_conf().output_bns_size(), decode_conf.blob_size());
  FOR_RANGE(int32_t, i, 0, decode_conf.blob_size()) {
    Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(i));
    CHECK_EQ(GetDataType<T>::val, out_blob->data_type());
    const std::string& name = decode_conf.blob(i).name();
    EncodeType encode_type = decode_conf.blob(i).encode_type();
    if (out_blob->has_col_num_field()) {
      ReadColNumToOutBlob(out_blob, name, encode_type, record_blob,
                          &(status->max_col_id_));
    }
    if (out_blob->has_data_id_field()) {
      ReadDataIdToOutBlob(out_blob, record_blob, kernel_ctx.device_ctx);
    }
    ReadDataContentToOutBlob(out_blob, name, encode_type, record_blob,
                             status->cur_col_id_, kernel_ctx.device_ctx);
  }
}

template<typename T>
void DecodeOFRecordKernel<T>::ReadDataContentToOutBlob(
    Blob* out_blob, const std::string& name, EncodeType encode_type,
    RecordBlob<OFRecord>* record_blob, int32_t col_id, DeviceCtx* device_ctx) {
  int32_t i = 0;
  int64_t col_size = out_blob->shape().Count(1);
  record_blob->ForEachRecord([&](const OFRecord& record) {
    const Feature& feature = record.feature().at(name);
    auto info = GetDptrAndColNum(feature, encode_type, col_size);
    const void* dptr = info.first;
    int32_t col_num = info.second;
    if (col_id < col_num) {
      if (encode_type == kRaw) {
        dptr = dptr + col_id * col_size;
      } else if (encode_type == kJpeg) {
        dptr = *(dptr + col_id);
      } else {
        UNIMPLEMENTED();
      }
      Memcpy<DeviceType::kCPU>(device_ctx, out_blob->mut_dptr() + col_size * i,
                               dptr, col_size);
    } else {
      memset(out_blob->mut_dptr() + col_size * i, 0, col_size);
    }
    ++i;
  });
  memset(out_blob->mut_dptr() + col_size * i, 0,
         col_size * (JobDesc::Singleton()->SinglePieceSize() - i));
}

}  // namespace oneflow
