#ifndef ONEFLOW_CORE_REGISTER_OFBLOB_H_
#define ONEFLOW_CORE_REGISTER_OFBLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

class OfBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfBlob);
  OfBlob(DeviceCtx* device_ctx, Blob* blob) : device_ctx_(device_ctx), blob_(blob) {
    mem_case_.mutable_host_mem();
  }
  ~OfBlob() = default;

  int data_type() const { return blob_->data_type(); }
  size_t NumAxes() const { return blob_->shape().NumAxes(); }
  void CopyShapeTo(int64_t* ptr, int64_t num_axis) const;

#define DEFINE_COPIER(T, type_proto)                                               \
  void CopyToBuffer_##T(T* ptr, int64_t len) const { AutoMemCopyTo<T>(ptr, len); } \
  void CopyFromBuffer_##T(const T* ptr, int64_t len) const { AutoMemCopyFrom<T>(ptr, len); }

  OF_PP_FOR_EACH_TUPLE(DEFINE_COPIER, POD_DATA_TYPE_SEQ);

#undef DEFINE_COPIER

  std::string GetCopyToBufferFuncNmae() const {
    static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) {type_proto, "CopyToBuffer_" #type_cpp},
        OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
    };
    return data_type2func_name.at(data_type());
  }

  std::string GetCopyFromBufferFuncNmae() const {
    static const HashMap<int64_t, std::string> data_type2func_name{
#define DATA_TYPE_FUNC_NAME_PAIR(type_cpp, type_proto) {type_proto, "CopyFromBuffer_" #type_cpp},
        OF_PP_FOR_EACH_TUPLE(DATA_TYPE_FUNC_NAME_PAIR, POD_DATA_TYPE_SEQ)
#undef DATA_TYPE_FUNC_NAME_PAIR
    };
    return data_type2func_name.at(data_type());
  }

 private:
  template<typename T>
  void AutoMemCopyTo(T* ptr, int64_t len) const;
  template<typename T>
  void AutoMemCopyFrom(const T* ptr, int64_t len) const;

  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
};

inline void OfBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

template<typename T>
void OfBlob::AutoMemCopyTo(T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, ptr, blob_->dptr(), len * sizeof(T), mem_case_, blob_->mem_case());
}

template<typename T>
void OfBlob::AutoMemCopyFrom(const T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, blob_->mut_dptr(), ptr, len * sizeof(T), blob_->mem_case(), mem_case_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OFBLOB_H_
