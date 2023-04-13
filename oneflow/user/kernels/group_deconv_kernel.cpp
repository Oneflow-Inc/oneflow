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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/include/primitive/matmul.h"

namespace oneflow {

namespace {

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewDeconvTransATransBMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, true, true);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewDeconvTransANoTransBMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, true, false);
}

auto DeconvTransATransBMatmulPrimitiveExists() {
  return hob::make_custom("DeconvTransATransBMatmulPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewDeconvTransATransBMatmulPrimitive(&ctx).operator bool();
                          });
}

auto DeconvTransANoTransBMatmulPrimitiveExists() {
  return hob::make_custom("DeconvTransANoTransBMatmulPrimitiveExists",
                          [](const user_op::KernelRegContext& ctx) {
                            return NewDeconvTransANoTransBMatmulPrimitive(&ctx).operator bool();
                          });
}

template<typename T>
using Col2ImFunc = void (*)(const T* col_buf, const ShapeView& in_shape,
                            const ShapeView& weight_shape, const ShapeView& out_shape,
                            const int32_t* strides, const int32_t* dilation_rate,
                            const int32_t* padding_before, T* in_diff_ptr);

template<typename T>
T* GetImgMutDptr(user_op::Tensor* tensor, int64_t idx) {
  return tensor->mut_dptr<T>() + tensor->shape_view().Count(1) * idx;
}

template<typename T>
const T* GetImgDptr(const user_op::Tensor* tensor, int64_t idx) {
  return tensor->dptr<T>() + tensor->shape_view().Count(1) * idx;
}

size_t CalcElemNumOfColBuf(const ShapeView& out_shape, const ShapeView& weight_shape,
                           const int32_t idx_offset) {
  int64_t col_buf_elem_cnt = 1;
  int64_t ndims = out_shape.NumAxes() - 2;
  for (size_t i = 0; i != ndims + 1; ++i) { col_buf_elem_cnt *= weight_shape.At(i + 1); }
  for (size_t i = 0; i != ndims; ++i) { col_buf_elem_cnt *= out_shape.At(idx_offset + i); }
  return col_buf_elem_cnt;
}

template<typename T>
class ColBufWriter {
 public:
  ColBufWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size)
      : src_ptr_(src_ptr),
        dst_ptr_(dst_ptr),
        c_size_(c_size),
        id_size_(id_size),
        ih_size_(ih_size),
        iw_size_(iw_size),
        od_size_(od_size),
        oh_size_(oh_size),
        ow_size_(ow_size) {}
  virtual ~ColBufWriter() = default;
  virtual void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) = 0;
  virtual void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) = 0;
  virtual void InvalidDFunc() = 0;
  virtual void InvalidHFunc() = 0;
  virtual void InvalidWFunc() = 0;
  virtual void NextImCSize() = 0;

 protected:
  const T* src_ptr_;
  T* dst_ptr_;
  int64_t c_size_ = 0;
  int64_t id_size_ = 0;
  int64_t ih_size_ = 0;
  int64_t iw_size_ = 0;
  int64_t od_size_ = 0;
  int64_t oh_size_ = 0;
  int64_t ow_size_ = 0;
};

template<typename T>
class Col2ImWriter final : public ColBufWriter<T> {
 public:
  Col2ImWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size)
      : ColBufWriter<T>::ColBufWriter(src_ptr, dst_ptr, c_size, id_size, ih_size, iw_size, od_size,
                                      oh_size, ow_size) {}
  ~Col2ImWriter() = default;
  void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override {
    this->dst_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw * this->iw_size_ + c] +=
        *(this->src_ptr_++);
  }
  void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override {
    this->dst_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw] += *(this->src_ptr_++);
  }
  void InvalidDFunc() override { this->src_ptr_ += this->od_size_; }
  void InvalidHFunc() override { this->src_ptr_ += this->oh_size_; }
  void InvalidWFunc() override { this->src_ptr_ += this->ow_size_; }
  void NextImCSize() override { this->dst_ptr_ += this->c_size_; }
};

template<typename T>
using DHWValidFunc = void (ColBufWriter<T>::*)(int64_t c, int64_t kd, int64_t kh, int64_t kw);

template<typename T>
class ColBufUtil final {
 public:
  ColBufUtil(const ShapeView& in_shape, const ShapeView& out_shape, int32_t dhw_offset,
             const int32_t* strides, const int32_t* dilation_rate, const int32_t* padding_before,
             const int32_t id_num, const int32_t ih_num, const int32_t iw_num, const int32_t od_num,
             const int32_t oh_num, const int32_t ow_num)
      : strides_(strides),
        dilation_rate_(dilation_rate),
        padding_before_(padding_before),
        id_num_(id_num),
        ih_num_(ih_num),
        iw_num_(iw_num),
        od_num_(od_num),
        oh_num_(oh_num),
        ow_num_(ow_num) {
    if (dhw_offset == 2) {
      dhw_valid_func_ = &ColBufWriter<T>::CDHWWrite;
    } else {
      dhw_valid_func_ = &ColBufWriter<T>::DHWCWrite;
    }
  }
  void operator()(ColBufWriter<T>* col_buf_writer, int64_t c, int64_t kd, int64_t kh, int64_t kw) {
    int64_t id = kd * dilation_rate_[0] - padding_before_[0];
    FOR_RANGE(int64_t, od, 0, od_num_) {
      if (id < 0 || id >= id_num_) {
        col_buf_writer->InvalidDFunc();
      } else {
        int64_t ih = kh * dilation_rate_[1] - padding_before_[1];
        FOR_RANGE(int64_t, oh, 0, oh_num_) {
          if (ih < 0 || ih >= ih_num_) {
            col_buf_writer->InvalidHFunc();
          } else {
            int64_t iw = kw * dilation_rate_[2] - padding_before_[2];
            FOR_RANGE(int64_t, ow, 0, ow_num_) {
              if (iw < 0 || iw >= iw_num_) {
                col_buf_writer->InvalidWFunc();
              } else {
                (col_buf_writer->*dhw_valid_func_)(c, id, ih, iw);
              }
              iw += strides_[2];
            }
          }
          ih += strides_[1];
        }
      }
      id += strides_[0];
    }
  }

 private:
  const int32_t* strides_;
  const int32_t* dilation_rate_;
  const int32_t* padding_before_;
  DHWValidFunc<T> dhw_valid_func_;
  int64_t id_num_ = 0;
  int64_t ih_num_ = 0;
  int64_t iw_num_ = 0;
  int64_t od_num_ = 0;
  int64_t oh_num_ = 0;
  int64_t ow_num_ = 0;
};

template<typename T>
struct DeconvKernelUtil final {
 public:
  static void NCDHWCol2Im(const T* col_buf_ptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before,
                               in_shape.At(2), in_shape.At(3), in_shape.At(4), out_shape.At(2),
                               out_shape.At(3), out_shape.At(4));
    Col2ImWriter<T> col_buf_writer(col_buf_ptr, in_diff_ptr, in_shape.Count(2), in_shape.Count(3),
                                   in_shape.Count(4), 1, out_shape.Count(3), out_shape.Count(4), 1);
    DoNCDWHFunc(weight_shape, col_buf_util, &col_buf_writer);
  }

  static void NDHWCCol2Im(const T* col_buf_ptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before,
                               in_shape.At(2), in_shape.At(3), in_shape.At(4), out_shape.At(2),
                               out_shape.At(3), out_shape.At(4));
    Col2ImWriter<T> col_buf_writer(col_buf_ptr, in_diff_ptr, in_shape.Count(2), in_shape.Count(2),
                                   in_shape.Count(3), in_shape.Count(4), out_shape.Count(2, 4),
                                   out_shape.Count(3, 4), 1);
    DoNDWHCFunc(weight_shape, col_buf_util, &col_buf_writer);
  }

 private:
  static void DoNCDWHFunc(const ShapeView& weight_shape, ColBufUtil<T>& col_buf_util,
                          ColBufWriter<T>* col_buf_writer) {
    for (int64_t c = 0; c != weight_shape.At(1); col_buf_writer->NextImCSize(), ++c) {
      for (int64_t kd = 0; kd != weight_shape.At(2); ++kd) {
        for (int64_t kh = 0; kh != weight_shape.At(3); ++kh) {
          for (int64_t kw = 0; kw != weight_shape.At(4); ++kw) {
            col_buf_util(col_buf_writer, c, kd, kh, kw);
          }
        }
      }
    }
  }

  static void DoNDWHCFunc(const ShapeView& weight_shape, ColBufUtil<T>& col_buf_util,
                          ColBufWriter<T>* col_buf_writer) {
    for (int64_t kd = 0; kd != weight_shape.At(1); ++kd) {
      for (int64_t kh = 0; kh != weight_shape.At(2); ++kh) {
        for (int64_t kw = 0; kw != weight_shape.At(3); ++kw) {
          for (int64_t c = 0; c != weight_shape.At(4); ++c) {
            col_buf_util(col_buf_writer, c, kd, kh, kw);
          }
        }
      }
    }
  }
};

template<typename T>
struct DeconvOpKernelCache final : public user_op::OpKernelCache {
  Col2ImFunc<T> col2im_func_ = DeconvKernelUtil<T>::NCDHWCol2Im;
  ;

  Shape in_5d_shape_;
  Shape out_5d_shape_;
  Shape weight_5d_shape_;

  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> dilation_rate_3d_;
  std::vector<int32_t> padding_before_3d_;

  bool is_out_diff_need_trans_ = false;

  int32_t idx_offset_ = 0;
  bool is_dynamic_ = false;
  int32_t groups = 1;

  void Update(const ShapeView& x_shape, const ShapeView& out_shape) {
    auto Gen5DShape = [](const ShapeView& shape, int32_t idx_offset) -> Shape {
      DimVector ret_vec;
      shape.ToDimVector(&ret_vec);
      int32_t ndims = ret_vec.size() - 2;
      ret_vec.insert(ret_vec.begin() + idx_offset, 3 - ndims, 1);
      return Shape(ret_vec);
    };
    if (is_dynamic_) {
      Shape in_shape;
      in_5d_shape_ = Gen5DShape(x_shape, idx_offset_);
      out_5d_shape_ = Gen5DShape(out_shape, idx_offset_);
    }
  }
};

template<typename T>
std::shared_ptr<DeconvOpKernelCache<T>> CreateDeconvOpKernelCache(user_op::KernelCacheContext* ctx,
                                                                  const std::string& in_name,
                                                                  const std::string& out_name,
                                                                  const std::string& weight_name) {
  const auto& data_format = ctx->Attr<std::string>("data_format");

  std::shared_ptr<DeconvOpKernelCache<T>> state(new DeconvOpKernelCache<T>());
  if (data_format == "channels_first") {
    state->col2im_func_ = DeconvKernelUtil<T>::NCDHWCol2Im;
    state->is_out_diff_need_trans_ = false;
    state->idx_offset_ = 2;
  } else {
    state->col2im_func_ = DeconvKernelUtil<T>::NDHWCCol2Im;
    state->is_out_diff_need_trans_ = true;
    state->idx_offset_ = 1;
  }

  auto Gen5DShape = [](const Shape& shape, int32_t idx_offset) -> Shape {
    DimVector ret_vec(shape.dim_vec());
    int32_t ndims = ret_vec.size() - 2;
    ret_vec.insert(ret_vec.begin() + idx_offset, 3 - ndims, 1);
    return Shape(ret_vec);
  };
  state->groups = ctx->Attr<int32_t>("groups");

  state->in_5d_shape_ =
      Gen5DShape(ctx->TensorDesc4ArgNameAndIndex(in_name, 0)->shape(), state->idx_offset_);
  state->out_5d_shape_ =
      Gen5DShape(ctx->TensorDesc4ArgNameAndIndex(out_name, 0)->shape(), state->idx_offset_);
  state->weight_5d_shape_ =
      Gen5DShape(ctx->TensorDesc4ArgNameAndIndex(weight_name, 0)->shape(), state->idx_offset_);

  auto Gen3DVec = [](const std::vector<int32_t>& origin_vec) -> std::vector<int32_t> {
    std::vector<int32_t> ret_vec = origin_vec;
    ret_vec.insert(ret_vec.begin(), 3 - ret_vec.size(), 1);
    return ret_vec;
  };
  state->strides_3d_ = Gen3DVec(ctx->Attr<std::vector<int32_t>>("strides"));
  state->dilation_rate_3d_ = Gen3DVec(ctx->Attr<std::vector<int32_t>>("dilation_rate"));
  state->is_dynamic_ = ctx->TensorDesc4ArgNameAndIndex(in_name, 0)->is_dynamic();
  const auto& padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
  FOR_RANGE(uint8_t, dim, 0, 3) {
    int64_t index = static_cast<int64_t>(dim) - (3 - padding_before.size());
    if (index < 0) {
      state->padding_before_3d_.emplace_back(0);
    } else {
      state->padding_before_3d_.emplace_back(padding_before.at(index));
    }
  }

  return state;
}

template<typename T>
class DeconvCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvCpuKernel);
  DeconvCpuKernel() = default;
  ~DeconvCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    if (*cache_ptr != nullptr && (flag & user_op::OpKernelCache::kAttrNotChanged)) {
      auto deconv_cache = std::dynamic_pointer_cast<DeconvOpKernelCache<T>>(*cache_ptr);
      deconv_cache->Update(ctx->TensorDesc4ArgNameAndIndex("in", 0)->shape(),
                           ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape());
      return;
    }
    *cache_ptr = CreateDeconvOpKernelCache<T>(ctx, "out", "in", "weight");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto deconv_cache = dynamic_cast<const DeconvOpKernelCache<T>*>(cache);
    CHECK_NOTNULL(deconv_cache);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    int32_t idx_offset = deconv_cache->idx_offset_;
    const int32_t input_group_interval = in->shape_view().At(1) / deconv_cache->groups;
    const int32_t weight_group_interval = weight->shape_view().At(0) / deconv_cache->groups;
    const int32_t output_group_interval = out->shape_view().At(1) / deconv_cache->groups;
    const int32_t input_step = input_group_interval * in->shape_view().Count(2);
    const int32_t weight_step = weight_group_interval * weight->shape_view().Count(1);
    const int32_t output_step = output_group_interval * out->shape_view().Count(2);
    const int32_t m = deconv_cache->weight_5d_shape_.Count(1);
    const int32_t n = deconv_cache->out_5d_shape_.Count(idx_offset, idx_offset + 3);
    const int32_t k = deconv_cache->weight_5d_shape_.At(0) / deconv_cache->groups;

    Memset<DeviceType::kCPU>(ctx->stream(), out->mut_dptr<T>(), 0,
                             out->shape_view().elem_cnt() * sizeof(T));

    std::unique_ptr<ep::primitive::Matmul> matmul;
    if (deconv_cache->is_out_diff_need_trans_) {
      matmul = NewDeconvTransATransBMatmulPrimitive(ctx);
    } else {
      matmul = NewDeconvTransANoTransBMatmulPrimitive(ctx);
    }
    CHECK(matmul);

    FOR_RANGE(int64_t, i, 0, in->shape_view().At(0)) {
      const T* input_ptr = GetImgDptr<T>(in, i);
      const T* weight_ptr = weight->dptr<T>();
      T* output_ptr = GetImgMutDptr<T>(out, i);

      FOR_RANGE(int64_t, g, 0, deconv_cache->groups) {
        matmul->Launch(ctx->stream(),
                       m,  //  (co / groups) * kd * kh * kw
                       n,  //  od * oh * ow
                       k,  //  filter / groups
                       static_cast<T>(1), weight_ptr, input_ptr, static_cast<T>(0),
                       col_buf->mut_dptr<T>());

        // out = col2im(col_buf')
        deconv_cache->col2im_func_(
            col_buf->mut_dptr<T>(), ShapeView(deconv_cache->in_5d_shape_),
            ShapeView(deconv_cache->weight_5d_shape_), ShapeView(deconv_cache->out_5d_shape_),
            deconv_cache->strides_3d_.data(), deconv_cache->dilation_rate_3d_.data(),
            deconv_cache->padding_before_3d_.data(), output_ptr);
        input_ptr += input_step;
        weight_ptr += weight_step;
        output_ptr += output_step;
      }
    }
  }
};

#define REGISTER_DECONV_DATA_KERNEL(op_name, dtype)                                     \
  REGISTER_USER_KERNEL(#op_name)                                                        \
      .SetCreateFn<DeconvCpuKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobAttr<int32_t>("groups") > 1)                     \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && DeconvTransATransBMatmulPrimitiveExists()                     \
                       && DeconvTransANoTransBMatmulPrimitiveExists())                  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                     \
        size_t tmp_buffer_size = 0;                                                     \
        const auto& in_shape = ctx->InputTensorDesc("in", 0).shape();                   \
        const auto& weight_shape = ctx->InputTensorDesc("weight", 0).shape();           \
                                                                                        \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));          \
        tmp_buffer_size +=                                                              \
            CalcElemNumOfColBuf(in_shape, weight_shape, idx_offset) * sizeof(dtype);    \
        return tmp_buffer_size;                                                         \
      })

REGISTER_DECONV_DATA_KERNEL(deconv1d, float);
REGISTER_DECONV_DATA_KERNEL(deconv1d, double);
REGISTER_DECONV_DATA_KERNEL(deconv2d, float);
REGISTER_DECONV_DATA_KERNEL(deconv2d, double);
REGISTER_DECONV_DATA_KERNEL(deconv3d, float);
REGISTER_DECONV_DATA_KERNEL(deconv3d, double);

}  // namespace

}  // namespace oneflow
