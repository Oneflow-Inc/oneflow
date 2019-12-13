#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/common/data_type_seq.h"

#define DEFINE_COPIER(T, type_proto)                                                     \
  void OfBlob_CurTensorCopyToBuffer_##T(uint64_t of_blob_ptr, T* array, int size) {      \
    using namespace oneflow;                                                             \
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);                              \
    of_blob->CurTensorAutoMemCopyTo<T>(array, size);                                     \
  }                                                                                      \
  void OfBlob_CurMutTensorCopyFromBuffer_##T(uint64_t of_blob_ptr, T* array, int size) { \
    using namespace oneflow;                                                             \
    auto* of_blob = reinterpret_cast<OfBlob*>(of_blob_ptr);                              \
    of_blob->CurMutTensorAutoMemCopyFrom<T>(array, size);                                \
  }

OF_PP_FOR_EACH_TUPLE(DEFINE_COPIER, POD_DATA_TYPE_SEQ);

#undef DEFINE_COPIER

std::string Dtype_GetOfBlobCurTensorCopyToBufferFuncName(int64_t dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, "OfBlob_CurTensorCopyToBuffer_" #type_cpp},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return data_type2func_name.at(dtype);
}

std::string Dtype_GetOfBlobCurMutTensorCopyFromBufferFuncName(int64_t dtype) {
  using namespace oneflow;
  static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) \
  {type_proto, "OfBlob_CurMutTensorCopyFromBuffer_" #type_cpp},
      OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
  };
  return data_type2func_name.at(dtype);
}
