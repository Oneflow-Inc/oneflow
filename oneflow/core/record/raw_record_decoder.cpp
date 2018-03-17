#include "oneflow/core/record/raw_record_decoder.h"

namespace oneflow {

namespace {

template<typename InType, typename OutType>
void ReadFromInDptrToOutDptr(const InType* in_dptr, OutType* out_dptr,
                             int64_t size) {
  FOR_RANGE(int64_t, i, 0, size) {
    *(out_dptr++) = static_cast<OutType>(*(in_dptr++));
  }
}

}  // namespace

template<typename T>
int32_t RawRecordDecoder<T>::GetColNumOfFeature(const Feature& feature,
                                                int64_t item_size) {
  if (feature.has_bytes_list()) {
    const std::string& str_list = feature.bytes_list().value(0);
    return str_list.size() / item_size;
  } else {
    return SizeOf(feature) / item_size;
  }
}

template<typename T>
void RawRecordDecoder<T>::ReadDataContentForOneItem(const Feature& feature,
                                                    int32_t cur_col_id,
                                                    T* out_dptr,
                                                    int64_t item_size,
                                                    DeviceCtx* ctx) {
  if (GetDataType<T>::val == DataTypeOf(feature)) {
    const T* in_dptr = nullptr;
    if (feature.has_bytes_list()) {
      const std::string& str_list = feature.bytes_list().value(0);
      in_dptr =
          reinterpret_cast<const T*>(str_list.c_str()) + cur_col_id * item_size;
    } else if (feature.has_float_list()) {
      in_dptr = reinterpret_cast<const T*>(feature.float_list().value().data());
    } else if (feature.has_double_list()) {
      in_dptr =
          reinterpret_cast<const T*>(feature.double_list().value().data());
    } else if (feature.has_int32_list()) {
      in_dptr = reinterpret_cast<const T*>(feature.int32_list().value().data());
    } else {
      UNIMPLEMENTED();
    }
    Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size);
  } else {
    LOG(WARNING) << "Transform data from type " << DataTypeOf(feature) << " to "
                 << GetDataType<T>::val << ".";
    if (feature.has_float_list()) {
      const float* in_dptr =
          feature.float_list().value().data() + cur_col_id * item_size;
      ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
    } else if (feature.has_double_list()) {
      const double* in_dptr =
          feature.double_list().value().data() + cur_col_id * item_size;
      ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
    } else if (feature.has_int32_list()) {
      const int32_t* in_dptr =
          feature.int32_list().value().data() + cur_col_id * item_size;
      ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
    } else if (feature.has_bytes_list()) {
      const char* in_dptr = feature.bytes_list().value(0).c_str();
      ReadFromInDptrToOutDptr(in_dptr, out_dptr, item_size);
    } else {
      UNIMPLEMENTED();
    }
  }
}

template class RawRecordDecoder<int8_t>;
template class RawRecordDecoder<int32_t>;
template class RawRecordDecoder<float>;
template class RawRecordDecoder<double>;

}  // namespace oneflow
