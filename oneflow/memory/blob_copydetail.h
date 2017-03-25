#include "proto/oneflow.pb.h"
#include <google/protobuf/repeated_field.h>
#include <type_traits>

namespace oneflow {
namespace detail {
template <typename T>
inline void Copy(size_t n, const T* src, T* dst) {
  if (std::is_fundamental<T>::value) {
    memcpy(static_cast<void*>(dst), static_cast<const void*>(src), n*sizeof(T));
  }
  else {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
  }
}
// whether to support SrcType is different from DstType.
template <typename SrcType>
void CopyToProtoAsIs(
  const BlobProto::DataType& data_type,
  const size_t size,
  const SrcType* src,
  BlobProto& proto) {

  switch (data_type) {
  case BlobProto_DataType_FLOAT:
  {
    auto field = proto.mutable_float_data();
    field->Reserve(size);
    for (int i = 0; i < size; ++i) {
      field->Add(0);
    }
    Copy<SrcType>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
    break;
  }
  case BlobProto_DataType_INT32:
  {
    auto field = proto.mutable_int32_data();
    field->Reserve(size);
    for (int i = 0; i < size; ++i) {
      field->Add(0);
    }
    Copy<SrcType>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
    break;
  }
  case BlobProto_DataType_INT64:
  {
    auto field = proto.mutable_int64_data();
    field->Reserve(size);
    for (int i = 0; i < size; ++i) {
      field->Add(0);
    }
    Copy<SrcType>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
    break;
  }
  case BlobProto_DataType_DOUBLE:
  {
    auto field = proto.mutable_double_data();
    field->Reserve(size);
    for (int i = 0; i < size; ++i) {
      field->Add(0);
    }
    Copy<SrcType>(
      size, src, reinterpret_cast<SrcType*>(field->mutable_data()));
    break;
  }
  default:
    LOG(FATAL) << "Undefined Type.";
    break;
  }
}

template <typename DstType>
bool CopyFromProtoAsIs(
  const BlobProto::DataType& data_type,
  const size_t size,
  const BlobProto& proto,
  DstType* dst) {
  //static_assert(
  //  sizeof(SrcType) == sizeof(DstType),
  //  "The source type and dest type cannot be copied as-is.");
  // need to support more data types.
  switch (data_type) {
  case BlobProto_DataType_FLOAT:
  {
    auto field = proto.float_data();
    Copy<DstType>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
    break;
  }
  case BlobProto_DataType_INT32:
  {
    auto field = proto.int32_data();
    Copy<DstType>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
    break;
  }
  case BlobProto_DataType_INT64:
  {
    auto field = proto.int64_data();
    Copy<DstType>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
    break;
  }
  case BlobProto_DataType_DOUBLE:
  {
    auto field = proto.double_data();
    Copy<DstType>(
      size, reinterpret_cast<const DstType*>(field.data()), dst);
    break;
  }
  default:
    LOG(FATAL) << "Undefined Type.";
    return false;
  }
  return true;
}
}
}
