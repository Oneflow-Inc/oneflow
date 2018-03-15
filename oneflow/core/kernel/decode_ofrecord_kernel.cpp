#include "oneflow/core/kernel/decode_ofrecord_kernel.h"

namespace oneflow {

template<typename T>
void DecodeOFRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CHECK(kernel_ctx.other);
  auto status = static_cast<DecodeStatus*>(kernel_ctx.other);
  Blob* first_out_blob = BnInOp2Blob("out_0");
  CHECK_EQ(GetDataType<T>::val, first_out_blob->data_type());

  RecordBlob<OfRecord>* record_blob =
      status->in_regst_->GetRecordBlob<OfRecord>();
  const DecodeOFRecordOpConf& conf = op_conf().decode_ofrecord_conf();
  FOR_RANGE(int32_t, i, 0, conf.blob_size()) {
    Blob* out_blob = BnInOp2Blob("out_" + std::to_string(i));
    const std::string& name = conf.blob(i).name();
    if (out_blob->has_col_num_field()) {
      ReadColNumToOutBlob(out_blob, name, record_blob, status->max_col_id_);
    }
    if (out_blob->has_data_id_field()) {
      ReadDataIdToOutBlob(out_blob, record_blob, kernel_ctx.device_ctx);
    }
    ReadDataContentToOutBlob(out_blob, name, record_blob, status->cur_col_id_,
                             kernel_ctx.device_ctx);
  }
}

template<typename T>
std::pair<const void*, int64_t> GetValueListInfo(const T& feature_list) {
  return {static_cast<const void*>(feature_list.data()),
          feature_list.size() * sizeof(typename T::value_type)};
}

std::pair<const void*, int64_t> GetFeatureInfo(const Feature& feature) {
  if (feature.has_bytes_list()) {
    // only consider the Raw Type.
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    return GetValueListInfo(feature.bytes_list().value(0));
  } else if (feature.has_float_list()) {
    return GetValueListInfo(feature.float_list().value());
  } else if (feature.has_double_list()) {
    return GetValueListInfo(feature.double_list().value());
  } else if (feature.has_int32_list()) {
    return GetValueListInfo(feature.int32_list().value());
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
void DecodeOFRecordKernel<T>::ReadColNumToOutBlob(
    Blob* out_blob, const std::string& name, RecordBlob<OfRecord>* record_blob,
    int32_t& max_col_id) {
  CHECK(out_blob->has_col_num_field());
  int32_t i = 0;
  int64_t col_size = out_blob->shape().Count(1);
  record_blob->ForEachRecord([&](const OfRecord& record) {
    const Feature& feature = record.feature().at(name);
    int32_t col_num = GetFeatureInfo(feature).second / col_size;
    CHECK(col_num <= out_blob->max_col_num());
    max_col_id = std::max(max_col_id, col_num - 1);
    out_blob->set_col_num(i++, col_num);
  });
  while (i < JobDesc::Singleton()->SinglePieceSize()) {
    out_blob->set_col_num(i++, 0);
  }
}

template<typename T>
void DecodeOFRecordKernel<T>::ReadDataIdToOutBlob(
    Blob* out_blob, RecordBlob<OfRecord>* record_blob, DeviceCtx* device_ctx) {
  CHECK(out_blob->has_data_id_field());
  int32_t i = 0;
  size_t size_of_data_id = JobDesc::Singleton()->SizeOfOneDataId();
  record_blob->ForEachRecord([&](const OfRecord& record) {
    const Feature& feature = record.feature().at("data_id");
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& data_id_str = feature.bytes_list().value(0);
    CHECK(data_id_str.size() <= size_of_data_id);
    Memcpy<DeviceType::kCPU>(device_ctx, out_blob->mut_data_id(i++),
                             &data_id_str, data_id_str.size());
  });
  memset(out_blob->mut_data_id(i), 0,
         size_of_data_id * (JobDesc::Singleton()->SinglePieceSize() - i));
}

template<typename T>
void DecodeOFRecordKernel<T>::ReadDataContentToOutBlob(
    Blob* out_blob, const std::string& name, RecordBlob<OfRecord>* record_blob,
    int32_t col_id, DeviceCtx* device_ctx) {
  int32_t i = 0;
  int64_t col_size = out_blob->shape().Count(1);
  record_blob->ForEachRecord([&](const OfRecord& record) {
    const Feature& feature = record.feature().at(name);
    std::pair<const void*, int64_t> info = GetFeatureInfo(feature);
    int32_t col_num = out_blob->has_col_num_field() ? out_blob->col_num(i) : 1;
    if (col_id < col_num) {
      Memcpy<DeviceType::kCPU>(device_ctx, out_blob->mut_dptr() + col_size * i,
                               info.first + col_id * col_size, col_size);
    } else {
      memset(out_blob->mut_dptr() + col_size * i, 0, col_size);
    }
    ++i;
  });
  memset(out_blob->mut_dptr() + col_size * i, 0,
         col_size * (JobDesc::Singleton()->SinglePieceSize() - i));
}
}  // namespace oneflow
