#include "oneflow/api/python/dlpack/dlpack.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device.h"

namespace oneflow {

Maybe<Symbol<Device>> ToOneFlowDevice(const DLDevice& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU: return JUST(Device::New("cpu"));
    case DLDeviceType::kDLCUDA: return JUST(Device::New("cuda", ctx.device_id));
    default: UNIMPLEMENTED_THEN_RETURN() << "Unsupported device type: " << ctx.device_type;
  }
}

Maybe<DataType> ToOneFlowDataType(const DLDataType& dtype) {
  DataType ofdtype = DataType::kInvalidDataType;
  CHECK_EQ_OR_RETURN(dtype.lanes, 1) << "OneFlow does not support lanes != 1";
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8: ofdtype = DataType::kUInt8; break;
        default:
          UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: " << dtype.code << dtype.bits;
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8: ofdtype = DataType::kInt8; break;
        case 16: ofdtype = DataType::kInt16; break;
        case 32: ofdtype = DataType::kInt32; break;
        case 64: ofdtype = DataType::kInt64; break;
        default:
          UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: " << dtype.code << dtype.bits;
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16: ofdtype = DataType::kFloat16; break;
        case 32: ofdtype = DataType::kFloat; break;
        case 64: ofdtype = DataType::kDouble; break;
        default:
          UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: " << dtype.code << dtype.bits;
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16: ofdtype = DataType::kBFloat16; break;
        default: UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: bfloat" << dtype.bits;
      }
      break;
    case DLDataTypeCode::kDLComplex:
      UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: complex" << dtype.bits;
      break;
    default: UNIMPLEMENTED_THEN_RETURN() << "Unsupported code " << dtype.code;
  }
  CHECK_NE_OR_RETURN(ofdtype, DataType::kInvalidDataType);
  return ofdtype;
}

Maybe<one::Tensor> fromDLPack(const DLManagedTensor* src) {
  using namespace one;
  const auto& dl_tensor = src->dl_tensor;

  Symbol<Device> device = JUST(ToOneFlowDevice(dl_tensor.device));
  DataType dtype = JUST(ToOneFlowDataType(dl_tensor.dtype));

  // Build TensorMeta
  const Shape shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
  Symbol<LocalTensorMeta> tensor_meta;
  if (dl_tensor.strides) {
    const auto stride = Stride(dl_tensor.strides, dl_tensor.strides + dl_tensor.ndim);
    tensor_meta = SymbolOf(LocalTensorMeta(shape, stride, dtype, device));
  } else {
    tensor_meta = SymbolOf(LocalTensorMeta(shape, dtype, device));
  }

  // Build TensorBuffer
  const auto& Free = [src](char* dptr) {
    if (src->deleter) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      src->deleter(const_cast<DLManagedTensor*>(src));
    }
  };

  size_t array_size_in_bytes = shape.elem_cnt() * GetSizeOfDataType(dtype);
  auto tensor_data = std::make_shared<vm::OutsideVmTensorStorage>();
  tensor_data->set_blob_dptr(
      std::unique_ptr<char, std::function<void(char*)>>(static_cast<char*>(dl_tensor.data), Free),
      array_size_in_bytes);

  // Build TensorStorage: decrease ndarray reference count before releasing
  auto tensor_storage = std::make_shared<TensorStorage>(tensor_data);

  // Build Tensor
  auto tensor_impl = std::make_shared<EagerLocalTensorImpl>(tensor_storage,
                                                            /*requires_grad=*/false,
                                                            /*ls_leaf=*/true);

  // Init blob
  JUST(tensor_impl->InitEagerBlobObject(tensor_meta, NewLocalDepObject()));
  const auto& stream = JUST(GetDefaultStreamByDevice(device));
  const auto& eager_blob_object = JUST(tensor_impl->eager_blob_object());
  JUST(eager_blob_object->init_producer_stream(stream));
  eager_blob_object->set_last_used_stream(stream);
  std::shared_ptr<Tensor> out(new LocalTensor(tensor_impl));
  return out;
}

}  // namespace oneflow
