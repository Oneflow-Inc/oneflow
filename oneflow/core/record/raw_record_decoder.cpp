#include "oneflow/core/record/raw_record_decoder.h"

namespace oneflow {

template<typename T>
int32_t RawRecordDecoder<T>::GetColNumOfFeature(const Feature& feature,
                                                int64_t item_size) {
  FeatureListHandler* handler = GetFeatureListHandler(feature);
  if (feature.has_bytes_list()) {
    // seems strange to handle byteslist
    const std::string* str_list =
        *static_cast<const std::string**>(handler->DptrOf(feature));
    return str_list->size() / item_size;
  } else {
    return handler->SizeOf(feature) / item_size;
  }
}

template<typename T>
void RawRecordDecoder<T>::ReadDataContentForOneItem(T* out_dptr,
                                                    const Feature& feature,
                                                    int64_t item_size,
                                                    DeviceCtx* ctx) {
  FeatureListHandler* handler = GetFeatureListHandler(feature);
  if (GetDataType<T>::val == handler->data_type()) {
    const T* in_dptr = static_cast<const T*>(handler->DptrOf(feature));
    Memcpy<DeviceType::kCPU>(ctx, out_dptr, in_dptr, item_size);
  } else {
    LOG(WARNING) << "Transform data from type " << handler->data_type()
                 << " to " << GetDataType<T>::val << ".";
    if (handler->data_type() == DataType::kFloat) {
      auto in_dptr = static_cast<const float*>(handler->DptrOf(feature));
      FOR_RANGE(int64_t, i, 0, item_size) {
        *(out_dptr++) = static_cast<T>(*(in_dptr++));
      }
    } else if (handler->data_type() == DataType::kDouble) {
      auto in_dptr = static_cast<const double*>(handler->DptrOf(feature));
      FOR_RANGE(int64_t, i, 0, item_size) {
        *(out_dptr++) = static_cast<T>(*(in_dptr++));
      }
    } else if (handler->data_type() == DataType::kInt32) {
      auto in_dptr = static_cast<const int32_t*>(handler->DptrOf(feature));
      FOR_RANGE(int64_t, i, 0, item_size) {
        *(out_dptr++) = static_cast<T>(*(in_dptr++));
      }
    } else if (handler->data_type() == DataType::kInt8) {
      auto in_dptr =
          *(static_cast<const std::string**>(handler->DptrOf(feature)));
      FOR_RANGE(int64_t, i, 0, item_size) {
        *(out_dptr++) = static_cast<T>(in_dptr->at(i));
      }
    }
  }
}

}  // namespace oneflow
