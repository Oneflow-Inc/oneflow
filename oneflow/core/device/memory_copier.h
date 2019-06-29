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

class MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryCopier);
  MemoryCopier() = default;
  virtual ~MemoryCopier() = default;

  virtual void Copy(DeviceCtx* ctx, void* dst, const void* src, const MemoryCopyNdDesc& desc) const;

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
};

#ifdef WITH_CUDA

class CudaAsyncMemoryCopier final : public MemoryCopier {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaAsyncMemoryCopier);
  CudaAsyncMemoryCopier() = default;
  ~CudaAsyncMemoryCopier() override = default;

 private:
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

#define REGISTER_DEFAULT_MEMORY_COPIER(device_type, func)         \
  REGISTER_CLASS_CREATOR(device_type, DefaultMemoryCopierCreator, \
                         ([] { return new DefaultMemoryCopierCreator(func); }))

MemoryCopier* NewDefaultMemoryCopier(DeviceType device_type);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_MEMORY_COPIER_H_
