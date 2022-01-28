#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>

namespace oneflow{

namespace{

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
    return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename Context>
ep::primitive::BlasTransposeType GetBlasTransposeType(Context* ctx, const std::string& attr) {
    return GetBlasTransposeType(ctx->template Attr<bool>(attr));
}


Optional<cudaDataType_t> OptCudaDataType(DataType data_type) {
    switch (data_type) {
    case kFloat: return CUDA_R_32F;
    case kDouble: return CUDA_R_64F;
    case kFloat16: return CUDA_R_16F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUDA_R_16BF;
#endif  // CUDA_VERSION >= 11000
    default: return NullOpt;
    }
}

cudaDataType_t GetCudaDataType(DataType data_type) {
    auto cuda_data_type = OptCudaDataType(data_type);
    CHECK(cuda_data_type.has_value());
    return cuda_data_type.value_or(CUDA_R_32F);
}

union CublasScalarParameter {
    double d;
    float s;
};

CublasScalarParameter GetCublasScalarParameter(Scalar scalar, cudaDataType_t compute_type) {
    CublasScalarParameter sp{};
    if (compute_type == CUDA_R_64F) {
    sp.d = scalar.Value<double>();
    } else if (compute_type == CUDA_R_32F) {
    sp.s = scalar.Value<float>();
    } else {
    UNIMPLEMENTED();
    }
    return sp;
}

cublasComputeType_t GetComputeType(DataType data_type) {
    switch (data_type) {
    case kFloat: return CUBLAS_COMPUTE_32F;
    case kDouble: return CUBLAS_COMPUTE_64F;
    case kFloat16: return CUBLAS_COMPUTE_16F;
#if CUDA_VERSION >= 11000
    case kBFloat16: return CUBLAS_COMPUTE_32F_FAST_16BF;
#endif  // CUDA_VERSION >= 11000
    default: UNIMPLEMENTED(); return CUBLAS_COMPUTE_32F;
    }
}

void InferMatmulMNK(const ShapeView& a_shape, const ShapeView& b_shape, const ShapeView& c_shape,
                    ep::primitive::BlasTransposeType transpose_a, ep::primitive::BlasTransposeType transpose_b, size_t* m, size_t* n, size_t* k) {
    const int64_t num_a_axes = a_shape.NumAxes();
    CHECK_GE(num_a_axes, 2);
    const int64_t num_b_axes = b_shape.NumAxes();
    CHECK_GE(num_b_axes, 2);
    const int64_t num_c_axes = c_shape.NumAxes();
    CHECK_GE(num_c_axes, 2);
    if (transpose_a == ep::primitive::BlasTransposeType::N) {
    *m = a_shape.At(num_a_axes - 2);
    *k = a_shape.At(num_a_axes - 1);
    } else if (transpose_a == ep::primitive::BlasTransposeType::T) {
    *m = a_shape.At(num_a_axes - 1);
    *k = a_shape.At(num_a_axes - 2);
    } else {
    UNIMPLEMENTED();
    }
    if (transpose_b == ep::primitive::BlasTransposeType::N) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), *k);
    *n = b_shape.At(num_b_axes - 1);
    } else if (transpose_b == ep::primitive::BlasTransposeType::T) {
    CHECK_EQ(b_shape.At(num_b_axes - 1), *k);
    *n = b_shape.At(num_b_axes - 2);
    } else {
    UNIMPLEMENTED();
    }
    CHECK_EQ(c_shape.At(num_c_axes - 2), *m);
    CHECK_EQ(c_shape.At(num_c_axes - 1), *n);
}

// TODO: Use OpKernel State. 
// class FusedMatmulBiasAddReluKernelState final : public user_op::OpKernelState{
// public: 
//     explicit FusedMatmulBiasAddReluKernelState(user_op::KernelInitContext* ctx){
//         const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
//         const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");
//         size_t m = 0, n = 0, k = 0;
//         const auto a_shape = ctx->TensorDesc4ArgNameAndIndex("a", 0)->shape();
//         const auto b_shape = ctx->TensorDesc4ArgNameAndIndex("b", 0)->shape();
//         const auto c_shape = ctx->TensorDesc4ArgNameAndIndex("out", 0)->shape();
//         InferMatmulMNK(a_shape, b_shape, c_shape, trans_a, trans_b, &m, &n, &k);
        
//         const auto GetCublasOperation = [](ep::primitive::BlasTransposeType transpose_type) {
//             if (transpose_type == ep::primitive::BlasTransposeType::N) {
//               return CUBLAS_OP_N;
//             } else if (transpose_type == ep::primitive::BlasTransposeType::T) {
//               return CUBLAS_OP_T;
//             } else {
//               UNIMPLEMENTED();
//               return CUBLAS_OP_N;
//             }
//           };
        

//         OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(operationDesc_, cublas_dtype));
//         OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(cublas_a_desc_, cublas_dtype, n, k, cublas_ldb)); 
//         OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(cublas_b_desc_, cublas_dtype, k, m, cublas_lda)); 
//         OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(cublas_c_desc_, cublas_dtype, n, m, cublas_ldc)); 
//     }
//     cublasLtMatmulDesc_t* operationDesc_;
//     cublasLtMatrixLayout_t* cublas_b_desc_; 
//     cublasLtMatrixLayout_t* cublas_a_desc_;
//     cublasLtMatrixLayout_t* cublas_c_desc_;
// }; 

} // namespace

template<typename T>
class FusedMatmulBiasAddReluKernel final: public user_op::OpKernel{
public: 
    FusedMatmulBiasAddReluKernel() = default; 
    ~FusedMatmulBiasAddReluKernel() = default; 

    bool AlwaysComputeWhenAllOutputsEmpty() const override {return false; }

private: 
    using user_op::OpKernel::Compute;
    void Compute(user_op::KernelComputeContext* ctx) const override{
        const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0); 
        const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0); 
        const user_op::Tensor* cublas_a = b; 
        const user_op::Tensor* cublas_b = a; 

        const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0); 
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0); 
        const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("out", 0)->data_type();
        // TODO: Add check
        const T alpha = static_cast<T>(ctx->Attr<double>("alpha")); 
        const T beta = static_cast<T>(0.0); 

        const auto GetCublasOperation = [](ep::primitive::BlasTransposeType transpose_type) {
            if (transpose_type == ep::primitive::BlasTransposeType::N) {
              return CUBLAS_OP_N;
            } else if (transpose_type == ep::primitive::BlasTransposeType::T) {
              return CUBLAS_OP_T;
            } else {
              UNIMPLEMENTED();
              return CUBLAS_OP_N;
            }
          };
        
        const auto trans_a = GetBlasTransposeType(ctx, "transpose_a");
        const auto trans_b = GetBlasTransposeType(ctx, "transpose_b");

        size_t m = 0, n = 0, k = 0;
        InferMatmulMNK(a->shape(), b->shape(), out->shape(), trans_a, trans_b, &m, &n, &k);
        
        /*
        Matmul: A(m, k) x B(k, n) = C(m, n), it follows the row major. 
        In cublas, it use column major to compute, Bt(n, k) x At(k, m) = Ct(n, m). 
        And Ct matrix follows the column major is equal to C(m, n) which follows the row major.  
        */
        const size_t cublas_m = n;
        const size_t cublas_n = m;
        const size_t cublas_k = k;

        const cublasOperation_t cublas_trans_a = GetCublasOperation(trans_b);
        const cublasOperation_t cublas_trans_b = GetCublasOperation(trans_a);
        
        const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type); 
        const cudaDataType_t cuda_data_type = GetCudaDataType(data_type); 
        
        int cublas_lda = 0;
        if (trans_b == ep::primitive::BlasTransposeType::N) {
            cublas_lda = n;
        } else if (trans_b == ep::primitive::BlasTransposeType::T) {
            cublas_lda = k;
        } else {
            UNIMPLEMENTED();
        }

        int cublas_ldb = 0;
        if (trans_a == ep::primitive::BlasTransposeType::N) {
            cublas_ldb = k;
        } else if (trans_a == ep::primitive::BlasTransposeType::T) {
            cublas_ldb = m;
        } else {
            UNIMPLEMENTED();
        }
        const int cublas_ldc = n;

        #if CUDA_VERSION >= 11000
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
        #else
        cublasGemmAlgo_t algo =
            (data_type == DataType::kFloat16) ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
        #endif

        cublasLtMatmulDesc_t operationDesc = NULL;
        OF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_dtype, cuda_data_type));

        // For best performance when using the bias vector, specify beta == 0 and CUBLASLT_POINTER_MODE_HOST.(from https://docs.nvidia.com/cuda/cublas/index.html#cublasLtPointerMode_t)
        cublasLtPointerMode_t mode = CUBLASLT_POINTER_MODE_HOST; 
        OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
        
        OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &cublas_trans_a, sizeof(cublas_trans_a)));
        OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &cublas_trans_b, sizeof(cublas_trans_b)));
        
        // Set as matmul + bias_add + relu. 
        cublasLtEpilogue_t epilogue;
        epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
        OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))); 
        
        // Set bias ptr
        const T* bias_ptr = reinterpret_cast<const T*>(bias->dptr()); 
        OF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
        
        cublasLtMatrixLayout_t cublas_a_desc = NULL, cublas_b_desc = NULL, cublas_c_desc = NULL;
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_a_desc, cuda_data_type, cublas_trans_a == CUBLAS_OP_N ? cublas_m : cublas_k, cublas_trans_a == CUBLAS_OP_N ? cublas_k : cublas_m, cublas_lda));
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_b_desc, cuda_data_type, cublas_trans_b == CUBLAS_OP_N ? cublas_k : cublas_n, cublas_trans_b == CUBLAS_OP_N ? cublas_n : cublas_k, cublas_ldb)); 
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cublas_c_desc, cuda_data_type, cublas_m, cublas_n, cublas_ldc)); 

        OF_CUBLAS_CHECK(cublasLtMatmul(ctx->stream()->As<ep::CudaStream>()->cublas_lt_handle(),
                                       operationDesc,
                                       &alpha,
                                        reinterpret_cast<const T*>(cublas_a->dptr()),
                                        cublas_a_desc,
                                        reinterpret_cast<const T*>(cublas_b->dptr()),
                                        cublas_b_desc,
                                        &beta,
                                        reinterpret_cast<T*>(out->mut_dptr()),
                                        cublas_c_desc,
                                        reinterpret_cast<T*>(out->mut_dptr()),
                                        cublas_c_desc,
                                        NULL,
                                        NULL,
                                        0,
                                        0));
        OF_CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_a_desc)); 
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_b_desc)); 
        OF_CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cublas_c_desc)); 
    }
}; 

#define REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(cpp_type, data_type)                                     \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu").SetCreateFn<FusedMatmulBiasAddReluKernel<cpp_type>>().SetIsMatchedHob( \
    (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
    && (user_op::HobDataType("out", 0) == data_type));

REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(double, DataType::kDouble); 
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(float, DataType::kFloat); 
REGISTER_MATMUL_BIAS_ADD_RELU_KERNEL_GPU(half, DataType::kFloat16); 

} // namespace oneflow  