#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/ops/nn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/ops/nn_util.h"

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
  return GetImgMutDptr<T>(const_cast<user_op::Tensor*>(tensor), idx);
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

  Shape in_3d_shape_;
  Shape out_3d_shape_;
  Shape weight_3d_shape_;

  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> dilation_rate_3d_;
  std::vector<int32_t> padding_before_3d_;
};

std::vector<int32_t> Gen3DVec(const std::vector<int32_t>& origin_vec) {
  std::vector<int32_t> ret_vec = origin_vec;
  ret_vec.insert(ret_vec.begin(), 3 - ret_vec.size(), 1);
  return ret_vec;
}

Shape Gen3DShape(const Shape& shape) {
  DimVector ret_vec(shape.dim_vec());
  ret_vec.insert(ret_vec.begin(), 3 - ret_vec.size(), 1);
  return Shape(ret_vec);
}

template<typename T, size_t NDims>
class ConvCpuKernel final : public user_op::OpKernel {
 public:
  ConvCpuKernel() = default;
  ~ConvCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const {
    const auto& data_format = ctx->Attr<std::string>("data_format");

    std::shared_ptr<ConvOpKernelState<T>> state;
    if (data_format == "channels_first") {
      state->im2col_func_ = ConvKernelUtil<T>::NCDHWIm2Col;
      state->col2im_func_ = ConvKernelUtil<T>::NCDHWCol2Im;
      state->forward_func_ = Gemm4ChannelFirst;
    } else {
      state->im2col_func_ = ConvKernelUtil<T>::NDHWCIm2Col;
      state->col2im_func_ = ConvKernelUtil<T>::NDHWCCol2Im;
      state->forward_func_ = Gemm4ChannelLast;
    }
    state->in_3d_shape_ = Gen3DShape(ctx->TensorDesc4ArgNameAndIndex("in", 0)->shape());
    state->out_3d_shape_ = Gen3DShape(ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape());
    state->weight_3d_shape_ = Gen3DShape(ctx->TensorDesc4ArgNameAndIndex("weight", 0)->shape());
    state->strides_3d_ = Gen3DVec(ctx->Attr<std::vector<int32_t>>("strides"));
    state->dilation_rate_3d_ = Gen3DVec(ctx->Attr<std::vector<int32_t>>("dilation_rate"));
    {
      state->padding_before_3d_.resize(3);
      const auto& padding = ctx->Attr<std::string>("padding");
      for (int i = 0; i < 3; ++i) {
        CalcOutAndPadding(state->in_3d_shape_.At(i), state->weight_3d_shape_.At(i),
                          state->dilation_rate_3d_.at(i), state->strides_3d_.at(i), padding,
                          nullptr, &(state->padding_before_3d_.at(i)), nullptr);
      }
    }

    return std::move(state);
  }

  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int32_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));
    T* col_buf_dptr = tmp_buffer->mut_dptr<T>();

    auto DimRangeCount = [](const std::vector<int32_t>& vec, int begin, int end) -> int {
      if (begin >= end) { return 0; }
      int cnt = 1;
      for (int i = begin; i < end; ++i) { cnt *= vec.at(i); }
      return cnt;
    };

    auto* conv_state = dynamic_cast<ConvOpKernelState<T>*>(state);
    CHECK_NOTNULL(conv_state);
    for (int64_t i = 0; i < in->shape().At(0); ++i) {
      conv_state->im2col_func_(GetImgDptr<T>(in, i), ShapeView(conv_state->in_3d_shape_),
                               ShapeView(conv_state->weight_3d_shape_),
                               ShapeView(conv_state->out_3d_shape_), conv_state->strides_3d_.data(),
                               conv_state->dilation_rate_3d_.data(),
                               conv_state->padding_before_3d_.data(), col_buf_dptr);

      // channels first: out = weight * col_buf
      // channels last:  out = (weight * col_buf)(T)
      conv_state->forward_func_(
          CblasNoTrans, CblasNoTrans,
          conv_state->weight_3d_shape_.At(0),                           // filter
          conv_state->out_3d_shape_.Count(idx_offset, idx_offset + 3),  // od * oh * ow
          conv_state->weight_3d_shape_.Count(1),                        // ci * kd * kh * kw
          static_cast<T>(1), weight->dptr<T>(), col_buf_dptr, static_cast<T>(0),
          GetImgMutDptr<T>(out, i));

      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
      if (bias != nullptr) {
        int64_t num_of_col_buf = CalcElemNumOfColBuf(out->shape(), weight->shape(), idx_offset);
        int64_t num_of_bias_mul = tmp_buffer->shape().elem_cnt() - num_of_col_buf;
        CHECK_GT(num_of_bias_mul, 0);
        T* bias_mul_dptr = col_buf_dptr + num_of_col_buf * sizeof(T);

        // channels first:  out += bias * bias_mul
        // channels last:   out += (bias * bias_mul)(T)
        conv_state->forward_func_(
            CblasNoTrans, CblasNoTrans,
            conv_state->weight_3d_shape_.At(0),                           // filter
            conv_state->out_3d_shape_.Count(idx_offset, idx_offset + 3),  // od * oh * ow
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
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                          \
        return ctx.device_type() == DeviceType::kGPU                                        \
               && ctx.TensorDesc4ArgNameAndIndex("in", 0)->data_type()                      \
                      == GetDataType<dtype>::value;                                         \
      })                                                                                    \
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

}  // namespace

}  // namespace oneflow
