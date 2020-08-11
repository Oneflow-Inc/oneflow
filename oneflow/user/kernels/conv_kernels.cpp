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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
using Im2ColFunc = void (*)(const T* in_dptr, const ShapeView& in_shape,
                            const ShapeView& weight_shape, const ShapeView& out_shape,
                            const int32_t* strides, const int32_t* dilation_rate,
                            const int32_t* padding_before, T* col_buf);

template<typename T>
using Col2ImFunc = void (*)(const T* col_buf, const ShapeView& in_shape,
                            const ShapeView& weight_shape, const ShapeView& out_shape,
                            const int32_t* strides, const int32_t* dilation_rate,
                            const int32_t* padding_before, T* in_diff_ptr);

template<typename T>
using GemmFunc = void (*)(enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m,
                          const int n, const int k, const T alpha, const T* a, const T* b,
                          const T beta, T* c);

template<typename T>
void Gemm4ChannelFirst(enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const T alpha, const T* a, const T* b,
                       const T beta, T* c) {
  NewKernelUtil<DeviceType::kCPU>::OFGemm(nullptr, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

template<typename T>
void Gemm4ChannelLast(enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m,
                      const int n, const int k, const T alpha, const T* a, const T* b, const T beta,
                      T* c) {
  trans_a = (trans_a == CblasNoTrans) ? CblasTrans : CblasNoTrans;
  trans_b = (trans_b == CblasNoTrans) ? CblasTrans : CblasNoTrans;
  NewKernelUtil<DeviceType::kCPU>::OFGemm(nullptr, trans_b, trans_a, n, m, k, alpha, b, a, beta, c);
}

template<typename T>
T* GetImgMutDptr(user_op::Tensor* tensor, int64_t idx) {
  return tensor->mut_dptr<T>() + tensor->shape().Count(1) * idx;
}

template<typename T>
const T* GetImgDptr(const user_op::Tensor* tensor, int64_t idx) {
  return tensor->dptr<T>() + tensor->shape().Count(1) * idx;
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
  int64_t c_size_;
  int64_t id_size_;
  int64_t ih_size_;
  int64_t iw_size_;
  int64_t od_size_;
  int64_t oh_size_;
  int64_t ow_size_;
};

template<typename T>
class Im2ColWriter final : public ColBufWriter<T> {
 public:
  Im2ColWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size)
      : ColBufWriter<T>::ColBufWriter(src_ptr, dst_ptr, c_size, id_size, ih_size, iw_size, od_size,
                                      oh_size, ow_size) {}
  ~Im2ColWriter() = default;
  void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override {
    *(this->dst_ptr_++) =
        this->src_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw * this->iw_size_ + c];
  }
  void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override {
    *(this->dst_ptr_++) = this->src_ptr_[id * this->id_size_ + ih * this->ih_size_ + iw];
  }
  void InvalidDFunc() override {
    FOR_RANGE(int64_t, i, 0, this->od_size_) { *(this->dst_ptr_++) = 0; }
  }
  void InvalidHFunc() override {
    FOR_RANGE(int64_t, i, 0, this->oh_size_) { *(this->dst_ptr_++) = 0; }
  }
  void InvalidWFunc() override {
    FOR_RANGE(int64_t, i, 0, this->ow_size_) { *(this->dst_ptr_++) = 0; }
  }
  void NextImCSize() override { this->src_ptr_ += this->c_size_; }
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
             const int32_t* strides, const int32_t* dilation_rate, const int32_t* padding_before)
      : strides_(strides), dilation_rate_(dilation_rate), padding_before_(padding_before) {
    id_num_ = in_shape.At(dhw_offset);
    ih_num_ = in_shape.At(dhw_offset + 1);
    iw_num_ = in_shape.At(dhw_offset + 2);
    od_num_ = out_shape.At(dhw_offset);
    oh_num_ = out_shape.At(dhw_offset + 1);
    ow_num_ = out_shape.At(dhw_offset + 2);
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
  int64_t id_num_;
  int64_t ih_num_;
  int64_t iw_num_;
  int64_t od_num_;
  int64_t oh_num_;
  int64_t ow_num_;
  const int32_t* strides_;
  const int32_t* dilation_rate_;
  const int32_t* padding_before_;
  DHWValidFunc<T> dhw_valid_func_;
};

template<typename T>
struct ConvKernelUtil final {
 public:
  static void NCDHWIm2Col(const T* in_dptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* col_buf_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before);
    Im2ColWriter<T> col_buf_writer(in_dptr, col_buf_ptr, in_shape.Count(2), in_shape.Count(3),
                                   in_shape.Count(4), 1, out_shape.Count(3), out_shape.Count(4), 1);
    DoNCDWHFunc(weight_shape, col_buf_util, &col_buf_writer);
  }

  static void NDHWCIm2Col(const T* in_dptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* col_buf_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 1, strides, dilation_rate, padding_before);
    Im2ColWriter<T> col_buf_writer(in_dptr, col_buf_ptr, in_shape.Count(2), in_shape.Count(2),
                                   in_shape.Count(3), in_shape.Count(4), out_shape.Count(2, 4),
                                   out_shape.Count(3, 4), 1);
    DoNDWHCFunc(weight_shape, col_buf_util, &col_buf_writer);
  }

  static void NCDHWCol2Im(const T* col_buf_ptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 2, strides, dilation_rate, padding_before);
    Col2ImWriter<T> col_buf_writer(col_buf_ptr, in_diff_ptr, in_shape.Count(2), in_shape.Count(3),
                                   in_shape.Count(4), 1, out_shape.Count(3), out_shape.Count(4), 1);
    DoNCDWHFunc(weight_shape, col_buf_util, &col_buf_writer);
  }

  static void NDHWCCol2Im(const T* col_buf_ptr, const ShapeView& in_shape,
                          const ShapeView& weight_shape, const ShapeView& out_shape,
                          const int32_t* strides, const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr) {
    ColBufUtil<T> col_buf_util(in_shape, out_shape, 1, strides, dilation_rate, padding_before);
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
struct ConvOpKernelState final : public user_op::OpKernelState {
  Im2ColFunc<T> im2col_func_;
  Col2ImFunc<T> col2im_func_;
  GemmFunc<T> forward_func_;

  Shape in_5d_shape_;
  Shape out_5d_shape_;
  Shape weight_5d_shape_;

  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> dilation_rate_3d_;
  std::vector<int32_t> padding_before_3d_;

  enum CBLAS_TRANSPOSE is_out_diff_need_trans_;
  int32_t idx_offset_;
  bool is_dynamic_;

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
std::shared_ptr<user_op::OpKernelState> CreateConvOpKernelState(user_op::KernelInitContext* ctx,
                                                                const std::string& in_name,
                                                                const std::string& out_name,
                                                                const std::string& weight_name) {
  const auto& data_format = ctx->Attr<std::string>("data_format");

  std::shared_ptr<ConvOpKernelState<T>> state(new ConvOpKernelState<T>());
  if (data_format == "channels_first") {
    state->im2col_func_ = ConvKernelUtil<T>::NCDHWIm2Col;
    state->col2im_func_ = ConvKernelUtil<T>::NCDHWCol2Im;
    state->forward_func_ = Gemm4ChannelFirst;
    state->is_out_diff_need_trans_ = CblasNoTrans;
    state->idx_offset_ = 2;
  } else {
    state->im2col_func_ = ConvKernelUtil<T>::NDHWCIm2Col;
    state->col2im_func_ = ConvKernelUtil<T>::NDHWCCol2Im;
    state->forward_func_ = Gemm4ChannelLast;
    state->is_out_diff_need_trans_ = CblasTrans;
    state->idx_offset_ = 1;
  }

  auto Gen5DShape = [](const Shape& shape, int32_t idx_offset) -> Shape {
    DimVector ret_vec(shape.dim_vec());
    int32_t ndims = ret_vec.size() - 2;
    ret_vec.insert(ret_vec.begin() + idx_offset, 3 - ndims, 1);
    return Shape(ret_vec);
  };
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
      state->padding_before_3d_.push_back(0);
    } else {
      state->padding_before_3d_.push_back(padding_before.at(index));
    }
  }

  return std::move(state);
}

template<typename T>
void InitBiasMulBuf(T* dptr, int64_t num) {
  for (int64_t i = 0; i < num; ++i) { dptr[i] = 1; }
}

template<typename T, size_t NDims>
class ConvCpuKernel final : public user_op::OpKernel {
 public:
  ConvCpuKernel() = default;
  ~ConvCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    return CreateConvOpKernelState<T>(ctx, "in", "out", "weight");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    T* col_buf_dptr = tmp_buffer->mut_dptr<T>();

    auto* conv_state = dynamic_cast<ConvOpKernelState<T>*>(state);
    conv_state->Update(in->shape(), out->shape());
    CHECK_NOTNULL(conv_state);
    bool is_bias_mul_inited = false;
    for (int64_t i = 0; i < in->shape().At(0); ++i) {
      conv_state->im2col_func_(GetImgDptr<T>(in, i), ShapeView(conv_state->in_5d_shape_),
                               ShapeView(conv_state->weight_5d_shape_),
                               ShapeView(conv_state->out_5d_shape_), conv_state->strides_3d_.data(),
                               conv_state->dilation_rate_3d_.data(),
                               conv_state->padding_before_3d_.data(), col_buf_dptr);

      // channels first: out = weight * col_buf
      // channels last:  out = (weight * col_buf)(T)
      int32_t idx_offset = conv_state->idx_offset_;
      conv_state->forward_func_(
          CblasNoTrans, CblasNoTrans,
          conv_state->weight_5d_shape_.At(0),                           // filter
          conv_state->out_5d_shape_.Count(idx_offset, idx_offset + 3),  // od * oh * ow
          conv_state->weight_5d_shape_.Count(1),                        // ci * kd * kh * kw
          static_cast<T>(1), weight->dptr<T>(), col_buf_dptr, static_cast<T>(0),
          GetImgMutDptr<T>(out, i));

      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      if (bias != nullptr) {
        int64_t num_of_col_buf = CalcElemNumOfColBuf(out->shape(), weight->shape(), idx_offset);
        int64_t num_of_bias_mul = tmp_buffer->shape().elem_cnt() - num_of_col_buf;
        CHECK_GT(num_of_bias_mul, 0);
        T* bias_mul_dptr = col_buf_dptr + num_of_col_buf * sizeof(T);
        if (!is_bias_mul_inited) {
          InitBiasMulBuf(bias_mul_dptr, num_of_bias_mul);
          is_bias_mul_inited = true;
        }

        // channels first:  out += bias * bias_mul
        // channels last:   out += (bias * bias_mul)(T)
        conv_state->forward_func_(
            CblasNoTrans, CblasNoTrans,
            conv_state->weight_5d_shape_.At(0),                           // filter
            conv_state->out_5d_shape_.Count(idx_offset, idx_offset + 3),  // od * oh * ow
            1,                                                            // 1
            static_cast<T>(1), bias->dptr<T>(), bias_mul_dptr, static_cast<T>(1),
            GetImgMutDptr<T>(out, i));
      }
    }
  }
};

#define REGISTER_CONV_KERNEL(op_name, dtype, ndims)                                         \
  REGISTER_USER_KERNEL(#op_name)                                                            \
      .SetCreateFn<ConvCpuKernel<dtype, ndims>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                   \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                         \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                         \
        size_t tmp_buffer_size = 0;                                                         \
        const auto& out_shape = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape();         \
        const auto& weight_shape = ctx->TensorDesc4ArgNameAndIndex("weight", 0)->shape();   \
                                                                                            \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));              \
        tmp_buffer_size +=                                                                  \
            CalcElemNumOfColBuf(out_shape, weight_shape, idx_offset) * sizeof(dtype);       \
                                                                                            \
        const auto* bias = ctx->TensorDesc4ArgNameAndIndex("bias", 0);                      \
        if (bias != nullptr) {                                                              \
          int64_t bias_mul_cnt = 1;                                                         \
          for (int i = 0; i < ndims; ++i) { bias_mul_cnt *= out_shape.At(idx_offset + i); } \
          tmp_buffer_size += bias_mul_cnt * sizeof(dtype);                                  \
        }                                                                                   \
        return tmp_buffer_size;                                                             \
      })

REGISTER_CONV_KERNEL(conv1d, float, 1);
REGISTER_CONV_KERNEL(conv2d, float, 2);
REGISTER_CONV_KERNEL(conv3d, float, 3);
REGISTER_CONV_KERNEL(conv1d, double, 1);
REGISTER_CONV_KERNEL(conv2d, double, 2);
REGISTER_CONV_KERNEL(conv3d, double, 3);

template<typename T>
class ConvDataGradCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvDataGradCpuKernel);
  ConvDataGradCpuKernel() = default;
  ~ConvDataGradCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    return CreateConvOpKernelState<T>(ctx, "dx", "dy", "filter");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* conv_state = dynamic_cast<ConvOpKernelState<T>*>(state);
    CHECK_NOTNULL(conv_state);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* filter = ctx->Tensor4ArgNameAndIndex("filter", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    conv_state->Update(dx->shape(), dy->shape());
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx->mut_dptr<T>(), 0,
                             dx->shape().elem_cnt() * sizeof(T));

    int32_t idx_offset = conv_state->idx_offset_;
    FOR_RANGE(int64_t, i, 0, dy->shape().At(0)) {
      // channels first:  col_buf' = weight(T) * out[i]'
      // channels last :  col_buf' = weight(T) * out[i]'(T)
      NewKernelUtil<DeviceType::kCPU>::OFGemm(
          nullptr, CblasTrans, conv_state->is_out_diff_need_trans_,
          conv_state->weight_5d_shape_.Count(1),                        //  ci * kd * kh * kw
          conv_state->out_5d_shape_.Count(idx_offset, idx_offset + 3),  //  od * oh * ow
          conv_state->weight_5d_shape_.At(0),                           //  filter
          static_cast<T>(1), filter->dptr<T>(), GetImgDptr<T>(dy, i), static_cast<T>(0),
          col_buf->mut_dptr<T>());

      // in' = col2im(col_buf')
      conv_state->col2im_func_(col_buf->dptr<T>(), ShapeView(conv_state->in_5d_shape_),
                               ShapeView(conv_state->weight_5d_shape_),
                               ShapeView(conv_state->out_5d_shape_), conv_state->strides_3d_.data(),
                               conv_state->dilation_rate_3d_.data(),
                               conv_state->padding_before_3d_.data(), GetImgMutDptr<T>(dx, i));
    }
  }
};

#define REGISTER_CONV_DATA_GRAD_KERNEL(op_name, dtype)                                     \
  REGISTER_USER_KERNEL(#op_name)                                                           \
      .SetCreateFn<ConvDataGradCpuKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                  \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                        \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))     \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        size_t tmp_buffer_size = 0;                                                        \
        const auto& out_diff_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();    \
        const auto& weight_shape = ctx->TensorDesc4ArgNameAndIndex("filter", 0)->shape();  \
                                                                                           \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));             \
        tmp_buffer_size +=                                                                 \
            CalcElemNumOfColBuf(out_diff_shape, weight_shape, idx_offset) * sizeof(dtype); \
        return tmp_buffer_size;                                                            \
      })

REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, float);
REGISTER_CONV_DATA_GRAD_KERNEL(conv_data_grad, double);

template<typename T>
class ConvFilterGradCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvFilterGradCpuKernel);
  ConvFilterGradCpuKernel() = default;
  ~ConvFilterGradCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    return CreateConvOpKernelState<T>(ctx, "x", "dy", "filter_diff");
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* conv_state = dynamic_cast<ConvOpKernelState<T>*>(state);
    CHECK_NOTNULL(conv_state);

    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* filter_diff = ctx->Tensor4ArgNameAndIndex("filter_diff", 0);
    user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    conv_state->Update(x->shape(), dy->shape());

    Memset<DeviceType::kCPU>(ctx->device_ctx(), filter_diff->mut_dptr<T>(), 0,
                             filter_diff->shape().elem_cnt() * sizeof(T));
    int32_t idx_offset = conv_state->idx_offset_;
    FOR_RANGE(int64_t, i, 0, dy->shape().At(0)) {
      conv_state->im2col_func_(GetImgDptr<T>(x, i), ShapeView(conv_state->in_5d_shape_),
                               ShapeView(conv_state->weight_5d_shape_),
                               ShapeView(conv_state->out_5d_shape_), conv_state->strides_3d_.data(),
                               conv_state->dilation_rate_3d_.data(),
                               conv_state->padding_before_3d_.data(), col_buf->mut_dptr<T>());

      // channels first:  weight' += out[i]' * col_buf(T)
      // channels last :  weight' += out[i]'(T) * col_buf(T)
      NewKernelUtil<DeviceType::kCPU>::OFGemm(
          nullptr, conv_state->is_out_diff_need_trans_, CblasTrans,
          conv_state->weight_5d_shape_.At(0),                           //  filter
          conv_state->weight_5d_shape_.Count(1),                        //  ci * kd * kh * kw
          conv_state->out_5d_shape_.Count(idx_offset, idx_offset + 3),  //  od * oh * ow
          static_cast<T>(1), GetImgDptr<T>(dy, i), col_buf->dptr<T>(), static_cast<T>(1),
          filter_diff->mut_dptr<T>());
    }
  }
};

#define REGISTER_CONV_FILTER_GRAD_KERNEL(op_name, dtype)                                        \
  REGISTER_USER_KERNEL(#op_name)                                                                \
      .SetCreateFn<ConvFilterGradCpuKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                             \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))          \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                             \
        size_t tmp_buffer_size = 0;                                                             \
        const auto& out_diff_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();         \
        const auto& weight_diff_shape =                                                         \
            ctx->TensorDesc4ArgNameAndIndex("filter_diff", 0)->shape();                         \
                                                                                                \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));                  \
        tmp_buffer_size +=                                                                      \
            CalcElemNumOfColBuf(out_diff_shape, weight_diff_shape, idx_offset) * sizeof(dtype); \
        return tmp_buffer_size;                                                                 \
      })

REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, float);
REGISTER_CONV_FILTER_GRAD_KERNEL(conv_filter_grad, double);

template<typename T>
class ConvBiasGradCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBiasGradCpuKernel);
  ConvBiasGradCpuKernel() = default;
  ~ConvBiasGradCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* bias_diff = ctx->Tensor4ArgNameAndIndex("bias_diff", 0);
    user_op::Tensor* bias_mul_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    InitBiasMulBuf(bias_mul_buf->mut_dptr<T>(), bias_mul_buf->shape().elem_cnt());
    Memset<DeviceType::kCPU>(ctx->device_ctx(), bias_diff->mut_dptr<T>(), 0,
                             bias_diff->shape().elem_cnt() * sizeof(T));

    const auto& data_format = ctx->Attr<std::string>("data_format");
    int32_t idx_offset;
    enum CBLAS_TRANSPOSE is_out_diff_need_trans;
    int32_t filter;
    if (data_format == "channels_first") {
      idx_offset = 2;
      is_out_diff_need_trans = CblasNoTrans;
      filter = dy->shape().At(1);
    } else {
      idx_offset = 1;
      is_out_diff_need_trans = CblasTrans;
      filter = dy->shape().At(dy->shape().NumAxes() - 1);
    }
    int ndims = dy->shape().NumAxes() - 2;
    FOR_RANGE(int64_t, i, 0, dy->shape().At(0)) {
      // channels first:  bias' += out' * bias_mul
      // channels last:   bias' += out'(T) * bias_mul
      NewKernelUtil<DeviceType::kCPU>::OFGemm(
          nullptr, is_out_diff_need_trans, CblasNoTrans,
          filter,                                             //  filter
          1,                                                  //  1
          dy->shape().Count(idx_offset, idx_offset + ndims),  //  od * oh * ow
          static_cast<T>(1), GetImgDptr<T>(dy, i), bias_mul_buf->dptr<T>(), static_cast<T>(1),
          bias_diff->mut_dptr<T>());
    }
  }
};

#define REGISTER_CONV_BIAS_GRAD_KERNEL(op_name, dtype)                                         \
  REGISTER_USER_KERNEL(#op_name)                                                               \
      .SetCreateFn<ConvBiasGradCpuKernel<dtype>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                      \
                       & (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                            \
        const auto& out_diff_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();        \
        const int ndims = out_diff_shape.NumAxes() - 2;                                        \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));                 \
        int64_t bias_mul_cnt = 1;                                                              \
        for (int i = 0; i < ndims; ++i) { bias_mul_cnt *= out_diff_shape.At(idx_offset + i); } \
        return bias_mul_cnt * sizeof(dtype);                                                   \
      })

REGISTER_CONV_BIAS_GRAD_KERNEL(conv_bias_grad, float);
REGISTER_CONV_BIAS_GRAD_KERNEL(conv_bias_grad, double);
}  // namespace

}  // namespace oneflow
