/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/api/python/dlpack/dlpack.h"
#include "oneflow/api/python/exception/exception.h"
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/tensor_util.h"

namespace oneflow {

Maybe<Symbol<Device>> ToOneFlowDevice(const DLDevice& ctx) {
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU: return JUST(Device::New("cpu"));
#ifdef WITH_CUDA
    case DLDeviceType::kDLCUDA: return JUST(Device::New("cuda", ctx.device_id));
#endif
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
  auto tensor_data = std::make_shared<vm::TensorStorage>(false);
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
  return std::static_pointer_cast<Tensor>(std::make_shared<LocalTensor>(tensor_impl));
}

Maybe<DLDevice> ToDLDevice(Symbol<Device> ofdevice) {
  DLDevice ctx;
  ctx.device_id = ofdevice->device_id();
  switch (ofdevice->enum_type()) {
    case DeviceType::kCPU: ctx.device_type = DLDeviceType::kDLCPU; break;
#ifdef WITH_CUDA
    case DeviceType::kCUDA: ctx.device_type = DLDeviceType::kDLCUDA; break;
#endif
    default: UNIMPLEMENTED_THEN_RETURN() << "Unsupported device type: " << ofdevice->type();
  }
  return ctx;
}

Maybe<DLDataType> ToDLDataType(DataType ofdtype) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = GetSizeOfDataType(ofdtype) * 8;
  switch (ofdtype) {
    case DataType::kUInt8: dtype.code = DLDataTypeCode::kDLUInt; break;
    case DataType::kInt8: dtype.code = DLDataTypeCode::kDLInt; break;
    case DataType::kInt16: dtype.code = DLDataTypeCode::kDLInt; break;
    case DataType::kInt32: dtype.code = DLDataTypeCode::kDLInt; break;
    case DataType::kInt64: dtype.code = DLDataTypeCode::kDLInt; break;
    case DataType::kFloat16: dtype.code = DLDataTypeCode::kDLFloat; break;
    case DataType::kFloat: dtype.code = DLDataTypeCode::kDLFloat; break;
    case DataType::kDouble: dtype.code = DLDataTypeCode::kDLFloat; break;
    case DataType::kBFloat16: dtype.code = DLDataTypeCode::kDLBfloat; break;
    default: UNIMPLEMENTED_THEN_RETURN() << "Unsupported data type: " << DataType_Name(ofdtype);
  }
  return dtype;
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct ATenDLMTensor {
  std::shared_ptr<one::Tensor> handle;
  DLManagedTensor tensor;
};

void deleter(DLManagedTensor* arg) { delete static_cast<ATenDLMTensor*>(arg->manager_ctx); }

Maybe<DLManagedTensor*> toDLPack(const std::shared_ptr<one::Tensor>& src) {
  auto shape = *src->shape();
  auto strides = *JUST(src->stride());
  // create a new tensor with possibly normalized strides
  // Reference:
  // https://github.com/pytorch/pytorch/issues/83069
  // https://github.com/pytorch/pytorch/issues/82610
  for (int i = 0; i < src->ndim(); i++) {
    if (shape[i] <= 1) { strides[i] = 1; }
  }

  ATenDLMTensor* atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter;
  JUST(one::SyncAccessTensorWithTimeOut(
      src,
      [&](ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>& tensor) {
        atDLMTensor->tensor.dl_tensor.data = tensor->mut_raw_dptr();
      },
      "const"));
  auto dldevice = JUST(ToDLDevice(JUST(src->device())));
  auto dldtype = JUST(ToDLDataType(src->dtype()->data_type()));
  atDLMTensor->tensor.dl_tensor.device = *dldevice;
  atDLMTensor->tensor.dl_tensor.ndim = src->ndim();
  atDLMTensor->tensor.dl_tensor.dtype = *dldtype;
  atDLMTensor->tensor.dl_tensor.shape =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<int64_t*>(src->shape()->data());
  atDLMTensor->tensor.dl_tensor.strides =
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<int64_t*>(JUST(src->stride())->data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}

// This function is mostly copied from PyTorch
void DLPack_Capsule_Destructor(PyObject* data) {
  if (likely(!PyCapsule_IsValid(data, "dltensor"))) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }
  HANDLE_ERRORS
  // Causes overheads for validity checks again, but this case is rare
  // since consuming libraries should rename the capsule according to spec.
  // Note that this cannot set a python error (we checked validity above),
  // so we don't need to handle python error state here.
  DLManagedTensor* dlMTensor = (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // the dlMTensor has not been consumed, call deleter ourselves.
  // DLPack spec mentions that deleter may be NULL, but deleter from
  // `flow.to_dlpack` is never NULL, so no need for an additional check here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
  END_HANDLE_ERRORS_RET()
}

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  m.def("to_dlpack", [](const std::shared_ptr<one::Tensor>& tensor) -> Maybe<py::capsule> {
    DLManagedTensor* dlMTensor = JUST(toDLPack(tensor));
    return py::capsule(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
  });
  // from_dlpack is exported in tensor_api.yaml
}

}  // namespace oneflow
