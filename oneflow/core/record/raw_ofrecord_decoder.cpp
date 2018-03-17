#include "oneflow/core/record/raw_ofrecord_decoder.h"

namespace oneflow {

namespace {

template<typename InType, typename OutType>
void ReadFromInDptrToOutDptr(const InType* in_dptr, OutType* out_dptr,
                             int64_t size) {
  FOR_RANGE(int64_t, i, 0, size) {
    *(out_dptr++) = static_cast<OutType>(*(in_dptr++));
  }
}

template<typename T, DataType D>
void ReadFromInDptrToOutBlob(const T* in_dptr, Blob* out_blob, int32_t col_id) {
  int64_t item_size = out_blob->shape().Count(1);
  in_dptr += col_id * item_size;
  if (D == DataType::kInt8) {
    int8_t* out_dptr = out_blob->mut_dptr<int8_t>() + col_id * item_size;
    ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
  } else if (D == DataType::kInt32) {
    int32_t* out_dptr = out_blob->mut_dptr<int32_t>() + col_id * item_size;
    ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
  } else if (D == DataType::kFloat) {
    float* out_dptr = out_blob->mut_dptr<float>() + col_id * item_size;
    ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
  } else if (D == DataType::kDouble) {
    double* out_dptr = out_blob->mut_dptr<double>() + col_id * item_size;
    ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
  }
}

}  // namespace

template<DataType D>
int32_t RawOFRecordDecoder<D>::GetColNumOfFeature(const Feature& feature,
                                                  int64_t item_size) {
  if (feature.has_bytes_list()) {
    const std::string& str_list = feature.bytes_list().value(0);
    return str_list.size() / item_size;
  } else {
    return SizeOf(feature) / item_size;
  }
}

template<DataType D>
void RawOFRecordDecoder<D>::ReadDataContentForOneItem(const Feature& feature,
                                                      int32_t col_id,
                                                      Blob* out_blob,
                                                      DeviceCtx* ctx) {
  if (D == DataTypeOf(feature)) {
    ReadWithSameDataType(feature, col_id, out_blob, ctx);
  } else {
    LOG(WARNING) << "Transform data from type " << DataTypeOf(feature) << " to "
                 << D << ".";
    if (feature.has_float_list()) {
      const float* in_dptr = feature.float_list().value().data();
      ReadFromInDptrToOutBlob<float, D>(in_dptr, out_blob, col_id);
    } else if (feature.has_double_list()) {
      const double* in_dptr = feature.double_list().value().data();
      ReadFromInDptrToOutBlob<double, D>(in_dptr, out_blob, col_id);
    } else if (feature.has_int32_list()) {
      const int32_t* in_dptr = feature.int32_list().value().data();
      ReadFromInDptrToOutBlob<int32_t, D>(in_dptr, out_blob, col_id);
    } else if (feature.has_bytes_list()) {
      const char* in_dptr = feature.bytes_list().value(0).c_str();
      ReadFromInDptrToOutBlob<char, D>(in_dptr, out_blob, col_id);
    } else {
      UNIMPLEMENTED();
    }
  }
}

template<>
void RawOFRecordDecoder<DataType::kInt8>::ReadWithSameDataType(
    const Feature& feature, int32_t col_id, Blob* out_blob, DeviceCtx* ctx) {
  int64_t item_size = out_blob->shape().Count(1);
  const int8_t* in_dptr =
      reinterpret_cast<const int8_t*>(feature.bytes_list().value(0).c_str())
      + col_id * item_size;
  int8_t* out_dptr = out_blob->mut_dptr<int8_t>() + col_id * item_size;
  Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size * sizeof(int8_t));
}

template<>
void RawOFRecordDecoder<DataType::kInt32>::ReadWithSameDataType(
    const Feature& feature, int32_t col_id, Blob* out_blob, DeviceCtx* ctx) {
  int64_t item_size = out_blob->shape().Count(1);
  const int32_t* in_dptr =
      feature.int32_list().value().data() + col_id * item_size;
  int32_t* out_dptr = out_blob->mut_dptr<int32_t>() + col_id * item_size;
  Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size * sizeof(int32_t));
}

template<>
void RawOFRecordDecoder<DataType::kFloat>::ReadWithSameDataType(
    const Feature& feature, int32_t col_id, Blob* out_blob, DeviceCtx* ctx) {
  int64_t item_size = out_blob->shape().Count(1);
  const float* in_dptr =
      feature.float_list().value().data() + col_id * item_size;
  float* out_dptr = out_blob->mut_dptr<float>() + col_id * item_size;
  Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size * sizeof(float));
}

template<>
void RawOFRecordDecoder<DataType::kDouble>::ReadWithSameDataType(
    const Feature& feature, int32_t col_id, Blob* out_blob, DeviceCtx* ctx) {
  int64_t item_size = out_blob->shape().Count(1);
  const double* in_dptr =
      feature.double_list().value().data() + col_id * item_size;
  double* out_dptr = out_blob->mut_dptr<double>() + col_id * item_size;
  Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size * sizeof(double));
}

template class RawOFRecordDecoder<DataType::kInt8>;
template class RawOFRecordDecoder<DataType::kInt32>;
template class RawOFRecordDecoder<DataType::kFloat>;
template class RawOFRecordDecoder<DataType::kDouble>;

}  // namespace oneflow
