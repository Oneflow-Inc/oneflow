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
#ifndef ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_
#define ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/nd_index.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

struct MemoryCopyNdDesc {
  Shape dst_shape;
  Shape src_shape;
  NdIndex dst_pos;
  NdIndex src_pos;
  Shape extent;

  MemoryCopyNdDesc CreateDimReducedDesc() const;
};

template<int32_t NDIMS>
void CopyNDCpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc);
#ifdef WITH_CUDA
template<int32_t NDIMS>
void CopyNDGpuImpl(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc);
#endif

class MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopier);
  MemoryCopier() = default;
  virtual ~MemoryCopier() = default;

  virtual void Copy(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) const;

  template<typename T>
  void CopyElem(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) const;

 protected:
  virtual void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const = 0;
  virtual void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src,
                      size_t src_pitch, size_t width, size_t height) const;
  virtual void Copy3D(DeviceCtx* ctx, void* dst, const void* src,
                      const MemoryCopyNdDesc& desc) const;
  virtual void CopyND(DeviceCtx* ctx, void* dst, const void* src,
                      const MemoryCopyNdDesc& desc) const;
};

class HostMemoryCopier final : public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HostMemoryCopier);
  HostMemoryCopier() = default;
  ~HostMemoryCopier() override = default;

 private:
  void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const override;
  void CopyND(DeviceCtx* ctx, void* dst, const void* src,
              const MemoryCopyNdDesc& desc) const override;
};

#ifdef WITH_CUDA

class CudaAsyncMemoryCopier final : public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAsyncMemoryCopier);
  CudaAsyncMemoryCopier() = default;
  ~CudaAsyncMemoryCopier() override = default;

 private:
  void Copy(DeviceCtx* ctx, void* dst, const void* src,
            const MemoryCopyNdDesc& desc) const override;
  void Copy1D(DeviceCtx* ctx, void* dst, const void* src, size_t count) const override;
  void Copy2D(DeviceCtx* ctx, void* dst, size_t dst_pitch, const void* src, size_t src_pitch,
              size_t width, size_t height) const override;
  void Copy3D(DeviceCtx* ctx, void* dst, const void* src,
              const MemoryCopyNdDesc& desc) const override;
  void CopyND(DeviceCtx* ctx, void* dst, const void* src,
              const MemoryCopyNdDesc& desc) const override;
};

#endif

class DefaultMemoryCopierCreator final {
 public:
  using Func = std::function<MemoryCopier*()>;
  OF_DISALLOW_COPY_AND_MOVE(DefaultMemoryCopierCreator)
  explicit DefaultMemoryCopierCreator(Func f) : func_(std::move(f)) {}
  ~DefaultMemoryCopierCreator() = default;

  MemoryCopier* Create() { return func_(); }

 private:
  const Func func_;
};

#define REGISTER_DEFAULT_MEMORY_COPIER(device_type, func)                  \
  REGISTER_CLASS_CREATOR(int32_t, device_type, DefaultMemoryCopierCreator, \
                         ([] { return new DefaultMemoryCopierCreator(func); }))

MemoryCopier* NewDefaultMemoryCopier(DeviceType device_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_
