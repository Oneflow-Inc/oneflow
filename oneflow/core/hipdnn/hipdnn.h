/*
 Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef HIPDNN_H
#define HIPDNN_H

#ifdef WITH_ROCM

#define HIPDNN_BN_MIN_EPSILON 1e-05

#include <hip/hip_runtime_api.h>

#define HIPDNN_VERSION 7000

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_HIP(expression)                                                   \
    {                                                                           \
        hipError_t error = (expression);                                        \
        if (error != hipSuccess) {                                              \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",                    \
                    hipGetErrorString(error), error, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

#define CHECK_HIPDNN(expression)                                                \
    {                                                                           \
        hipdnnStatus_t error = (expression);                                    \
        if (error != HIPDNN_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "HIPDNN error: %s (%d) at %s:%d\n",                 \
                    hipdnnGetErrorString(error), error, __FILE__, __LINE__);    \
            return error;                                                       \
        }                                                                       \
    }

#define CHECK_MALLOC(pointer)                                                   \
    {                                                                           \
        if ( (pointer) == 0) {   /*if Null pointer*/                            \
            fprintf(stderr, "Malloc failed error:%s:%d\n", __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    }

//============================ Datatypes =======================================

typedef enum {
    HIPDNN_STATUS_SUCCESS = 0,
    HIPDNN_STATUS_NOT_INITIALIZED = 1,
    HIPDNN_STATUS_ALLOC_FAILED = 2,
    HIPDNN_STATUS_BAD_PARAM = 3,
    HIPDNN_STATUS_INTERNAL_ERROR = 4,
    HIPDNN_STATUS_INVALID_VALUE = 5,
    HIPDNN_STATUS_ARCH_MISMATCH = 6,
    HIPDNN_STATUS_MAPPING_ERROR = 7,
    HIPDNN_STATUS_EXECUTION_FAILED = 8,
    HIPDNN_STATUS_NOT_SUPPORTED = 9,
    HIPDNN_STATUS_LICENSE_ERROR = 10,
    HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
} hipdnnStatus_t;

typedef enum {
    HIPDNN_DATA_FLOAT = 0,
    HIPDNN_DATA_DOUBLE = 1,
    HIPDNN_DATA_HALF = 2,
    HIPDNN_DATA_INT8 = 3,
    HIPDNN_DATA_INT32 = 4,
    HIPDNN_DATA_INT8x4 = 5
} hipdnnDataType_t;

typedef enum {
    HIPDNN_NOT_PROPAGATE_NAN = 0,
    HIPDNN_PROPAGATE_NAN = 1,
} hipdnnNanPropagation_t;

//------------------------- Tensors datatypes ----------------------------------

typedef enum {
    HIPDNN_DEFAULT_MATH = 0,
    HIPDNN_TENSOR_OP_MATH = 1,
} hipdnnMathType_t;

typedef enum {
    HIPDNN_TENSOR_NCHW = 0, /* row major (wStride = 1, hStride = w) */
    HIPDNN_TENSOR_NHWC = 1, /* feature maps interleaved ( cStride = 1 )*/
    HIPDNN_TENSOR_NCHW_VECT_C = 2 /* each image point is vector of element of C:
                          the length of the vector is carried by the data type*/
} hipdnnTensorFormat_t;

typedef enum {
    HIPDNN_OP_TENSOR_ADD = 0,
    HIPDNN_OP_TENSOR_MUL = 1,
    HIPDNN_OP_TENSOR_MIN = 2,
    HIPDNN_OP_TENSOR_MAX = 3,
    HIPDNN_OP_TENSOR_SQRT = 4,
    HIPDNN_OP_TENSOR_NOT  = 5,
} hipdnnOpTensorOp_t;

// CNTK 2.4

typedef enum {
    HIPDNN_REDUCE_TENSOR_ADD = 0,
    HIPDNN_REDUCE_TENSOR_MUL = 1,
    HIPDNN_REDUCE_TENSOR_MIN = 2,
    HIPDNN_REDUCE_TENSOR_MAX = 3,
    HIPDNN_REDUCE_TENSOR_AMAX = 4,
    HIPDNN_REDUCE_TENSOR_AVG = 5,
    HIPDNN_REDUCE_TENSOR_NORM1 = 6,
    HIPDNN_REDUCE_TENSOR_NORM2 = 7,
    HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
} hipdnnReduceTensorOp_t;

typedef enum {
    HIPDNN_REDUCE_TENSOR_NO_INDICES = 0,
    HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
} hipdnnReduceTensorIndices_t;

typedef enum {
    HIPDNN_32BIT_INDICES = 0,
    HIPDNN_64BIT_INDICES = 1,
    HIPDNN_16BIT_INDICES = 2,
    HIPDNN_8BIT_INDICES = 3,
} hipdnnIndicesType_t;

//------------------------- Convolutional datatypes ----------------------------

typedef enum {
    HIPDNN_CONVOLUTION = 0,         //Not supported in MIopen
    HIPDNN_CROSS_CORRELATION = 1,   //MIopen's Convolution
    HIPDNN_TRANSPOSE = 2,           //Not supported in CUDNN
    HIPDNN_GROUP_CONVOLUTION = 3,   //Not supported in CUDNN
    HIPDNN_DEPTHWISE = 4,           //Not supported in CUDNN
} hipdnnConvolutionMode_t;

typedef enum {
    HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    HIPDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    HIPDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    HIPDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
} hipdnnConvolutionFwdAlgo_t;

int ConvolutionFwdAlgoCount();

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(int i);


typedef enum {
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,  // non-deterministic
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3 =
        3,  // non-deterministic, algo0 with workspace
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,  // not implemented
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
    HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
} hipdnnConvolutionBwdFilterAlgo_t;

int ConvolutionBwdFilterAlgoCount();

// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t GetConvolutionBwdFilterAlgo(int i);


typedef enum {
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,  // non-deterministic
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM = 6,
    HIPDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 7,
} hipdnnConvolutionBwdDataAlgo_t;

int ConvolutionBwdDataAlgoCount();

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t GetConvolutionBwdDataAlgo(int i);


typedef enum {
    HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
    HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
    HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
} hipdnnConvolutionBwdDataPreference_t;

typedef enum {
    HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
    HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
    HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} hipdnnConvolutionFwdPreference_t;

typedef enum {
    HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
    HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
    HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
} hipdnnConvolutionBwdFilterPreference_t;

struct hipdnnConvolutionFwdAlgoPerf_t {
    hipdnnConvolutionFwdAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    hipdnnMathType_t mathType;
    long reserved[3];
};

struct hipdnnConvolutionBwdDataAlgoPerf_t {
    hipdnnConvolutionBwdDataAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    hipdnnMathType_t mathType;
    long reserved[3];
};

struct hipdnnConvolutionBwdFilterAlgoPerf_t {
    hipdnnConvolutionBwdFilterAlgo_t algo;
    hipdnnStatus_t status;
    float time;
    size_t memory;
    hipdnnMathType_t mathType;
    long reserved[3];
};

//-------------------------- Pooling datatypes ---------------------------------

typedef enum {
    HIPDNN_POOLING_MAX = 0,
    HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING =
        1,  // count for average includes padded values
    HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING =
        2,  // count for average does not include padded values
    HIPDNN_POOLING_MAX_DETERMINISTIC = 3
} hipdnnPoolingMode_t;

//--------------------------- LRN datatypes ------------------------------------

typedef enum {
    HIPDNN_LRN_WITHIN_CHANNEL = 0,
    HIPDNN_LRN_CROSS_CHANNEL = 1,
} hipdnnLRNMode_t;

//--------------------------- BN datatypes ------------------------------------

typedef enum {
    /* bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice,
     * normalized over N slice)
     */
    HIPDNN_BATCHNORM_PER_ACTIVATION = 0,

    /* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized
     * over Nx1xHxW subtensors)
     */
    HIPDNN_BATCHNORM_SPATIAL = 1,

    /* bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized
     * over Nx1xHxW subtensors). May be faster than CUDNN_BATCHNORM_SPATIAL but
     * imposes some limits on the range of values
     */
    HIPDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} hipdnnBatchNormMode_t;

//--------------------------- Activation datatypes -----------------------------

typedef enum {
    HIPDNN_ACTIVATION_SIGMOID = 0,
    HIPDNN_ACTIVATION_RELU,
    HIPDNN_ACTIVATION_TANH,
    HIPDNN_ACTIVATION_CLIPPED_RELU,
    HIPDNN_ACTIVATION_ELU,
    HIPDNN_ACTIVATION_PATHTRU,
    HIPDNN_ACTIVATION_SOFTRELU,
    HIPDNN_ACTIVATION_ABS,
    HIPDNN_ACTIVATION_POWER
} hipdnnActivationMode_t;

//--------------------------- Softmax datatypes -----------------------------

typedef enum {
    HIPDNN_SOFTMAX_FAST = 0, /* straightforward implementation */
    HIPDNN_SOFTMAX_ACCURATE =
        1, /* subtract max from every point to avoid overflow */
    HIPDNN_SOFTMAX_LOG = 2
} hipdnnSoftmaxAlgorithm_t;

typedef enum {
    HIPDNN_SOFTMAX_MODE_INSTANCE =
        0, /* compute the softmax over all C, H, W for each N */
    HIPDNN_SOFTMAX_MODE_CHANNEL =
        1 /* compute the softmax over all C for each H, W, N */
} hipdnnSoftmaxMode_t;

//--------------------------- RNN datatypes -----------------------------

typedef enum {
    HIPDNN_RNN_RELU = 0,  // Stock RNN with ReLu activation
    HIPDNN_RNN_TANH = 1,  // Stock RNN with tanh activation
    HIPDNN_LSTM = 2,      // LSTM with no peephole connections
    HIPDNN_GRU =
        3  // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z
           // * h(t-1);
} hipdnnRNNMode_t;

typedef enum {
    HIPDNN_UNIDIRECTIONAL = 0,
    HIPDNN_BIDIRECTIONAL = 1  // Using output concatination at each step.
                              // HGSOS: Do we also want to support output sum?
} hipdnnDirectionMode_t;

typedef enum {
    HIPDNN_LINEAR_INPUT = 0,
    HIPDNN_SKIP_INPUT = 1
} hipdnnRNNInputMode_t;

typedef enum {
    HIPDNN_RNN_ALGO_STANDARD = 0,
    HIPDNN_RNN_ALGO_PERSIST_STATIC = 1,
    HIPDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
} hipdnnRNNAlgo_t;

typedef enum {
    HIPDNN_RNN_NO_BIAS = 0,
    HIPDNN_RNN_WITH_BIAS = 1
} hipdnnRNNBiasMode_t;

typedef enum {
    HIPDNN_RNN_ALGO_GEMM = 0
}hipdnnRNNGEMMalgoMode_t;

//--------------------------- Fusion datatypes ---------------------------------

typedef enum {
    HIPDNN_VERTICAL_FUSION = 0,
    HIPDNN_HORIZONTAL_FUSION = 1,
} hipdnnFusionDirection_t;

//------------------------------- Opaque Pointers ------------------------------

typedef void *hipdnnHandle_t;

typedef void *hipdnnStream_t;

typedef void *hipdnnTensorDescriptor_t;

typedef void *hipdnnReduceTensorDescriptor_t;

typedef void *hipdnnFilterDescriptor_t;

typedef void *hipdnnConvolutionDescriptor_t;

typedef void *hipdnnLRNDescriptor_t;

typedef void *hipdnnActivationDescriptor_t;

typedef void *hipdnnPoolingDescriptor_t;

typedef void *hipdnnOpTensorDescriptor_t;

typedef void *hipdnnDropoutDescriptor_t;

typedef void *hipdnnRNNDescriptor_t;

typedef void *hipdnnPersistentRNNPlan_t;

typedef void *hipdnnDeterminism_t;

typedef void *hipdnnFusionPlanDescriptor_t;

typedef void *hipdnnFusionOpDescriptor_t;

typedef void *hipdnnOperatorArgs_t;

//==============================================================================

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle);

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle);

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId);

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle, hipdnnStream_t *streamId);

size_t hipdnnGetVersion(void);

//=============================== Tensors ======================================

hipdnnStatus_t
hipdnnCreateTensorDescriptor( hipdnnTensorDescriptor_t *tensorDesc);

hipdnnStatus_t
hipdnnSetTensor4dDescriptor( hipdnnTensorDescriptor_t tensorDesc,
                             hipdnnTensorFormat_t format,
                             hipdnnDataType_t dataType,
                             int n, int c, int h, int w);

hipdnnStatus_t
hipdnnGetTensor4dDescriptor( hipdnnTensorDescriptor_t tensorDesc,
                             hipdnnDataType_t *dataType,
                             int *n, int *c, int *h, int *w,
                             int *nStride, int *cStride,
                             int *hStride, int *wStride);

hipdnnStatus_t
hipdnnSetTensor4dDescriptorEx( hipdnnTensorDescriptor_t tensorDesc,
                               hipdnnDataType_t dataType, /* image data type */
                               int n,                     /* number of inputs (batch size) */
                               int c,                     /* number of input feature maps */
                               int h,                     /* height of input section */
                               int w,                     /* width of input section */
                               int nStride, int cStride, int hStride, int wStride);

hipdnnStatus_t
hipdnnSetTensorNdDescriptor( hipdnnTensorDescriptor_t tensorDesc,
                             hipdnnDataType_t dataType,
                             int nbDims,
                             const int dimA[],
                             const int strideA[]);

hipdnnStatus_t
hipdnnGetTensorNdDescriptor( const hipdnnTensorDescriptor_t tensorDesc,
                             int nbDimsRequested,
                             hipdnnDataType_t *dataType,
                             int *nbDims,
                             int dimA[],
                             int strideA[]);

hipdnnStatus_t
hipdnnDestroyTensorDescriptor( hipdnnTensorDescriptor_t tensorDesc);

hipdnnStatus_t hipdnnSetTensor( hipdnnHandle_t handle,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y,
                                const void *valuePtr );

hipdnnStatus_t hipdnnAddTensor( hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t aDesc,
                                const void *A,
                                const void *beta,
                                const hipdnnTensorDescriptor_t cDesc,
                                void *C);

hipdnnStatus_t hipdnnScaleTensor( hipdnnHandle_t handle,
                                  const hipdnnTensorDescriptor_t yDesc,
                                  void *y,
                                  const void *alpha);

//-------------------------- Tensor Operation -----------------------------------
hipdnnStatus_t
hipdnnCreateOpTensorDescriptor( hipdnnOpTensorDescriptor_t *opTensorDesc);

hipdnnStatus_t
hipdnnSetOpTensorDescriptor( hipdnnOpTensorDescriptor_t opTensorDesc,
                             hipdnnOpTensorOp_t opTensorOp,
                             hipdnnDataType_t opTensorCompType,
                             hipdnnNanPropagation_t opTensorNanOpt);

hipdnnStatus_t
hipdnnGetOpTensorDescriptor( const hipdnnOpTensorDescriptor_t opTensorDesc,
                             hipdnnOpTensorOp_t *opTensorOp,
                             hipdnnDataType_t *opTensorCompType,
                             hipdnnNanPropagation_t *opTensorNanOpt);

hipdnnStatus_t
hipdnnDestroyOpTensorDescriptor( hipdnnOpTensorDescriptor_t opTensorDesc);

hipdnnStatus_t hipdnnOpTensor( hipdnnHandle_t handle,
                               const hipdnnOpTensorDescriptor_t opTensorDesc,
                               const void *alpha1,
                               const hipdnnTensorDescriptor_t aDesc,
                               const void *A,
                               const void *alpha2,
                               const hipdnnTensorDescriptor_t bDesc,
                               const void *B,
                               const void *beta,
                               const hipdnnTensorDescriptor_t cDesc,
                               void *C);

//--------------------------- Tensor Reduction ---------------------------------

hipdnnStatus_t
hipdnnCreateReduceTensorDescriptor( hipdnnReduceTensorDescriptor_t *reduceTensorDesc);

hipdnnStatus_t
hipdnnSetReduceTensorDescriptor( hipdnnReduceTensorDescriptor_t reduceTensorDesc,
                                 hipdnnReduceTensorOp_t reduceTensorOp,
                                 hipdnnDataType_t reduceTensorCompType,
                                 hipdnnNanPropagation_t reduceTensorNanOpt,
                                 hipdnnReduceTensorIndices_t reduceTensorIndices,
                                 hipdnnIndicesType_t reduceTensorIndicesType);

hipdnnStatus_t
hipdnnGetReductionWorkspaceSize( hipdnnHandle_t handle,
                         const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
                         const hipdnnTensorDescriptor_t aDesc,
                         const hipdnnTensorDescriptor_t cDesc,
                         size_t *sizeInBytes);

hipdnnStatus_t
hipdnnReduceTensor( hipdnnHandle_t handle,
                    const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
                    void *indices,   size_t indicesSizeInBytes,
                    void *workspace, size_t workspaceSizeInBytes,
                    const void *alpha,
                    const hipdnnTensorDescriptor_t aDesc, const void *A,
                    const void *beta,
                    const hipdnnTensorDescriptor_t cDesc, void *C);

hipdnnStatus_t
hipdnnDestroyReduceTensorDescriptor( hipdnnReduceTensorDescriptor_t reduceTensorDesc);

//=============================== Filter =======================================
// In MIOpen Filter data and descriptor is same as tensor data and decriptor

hipdnnStatus_t
hipdnnCreateFilterDescriptor( hipdnnFilterDescriptor_t *filterDesc);

hipdnnStatus_t
hipdnnSetFilter4dDescriptor( hipdnnFilterDescriptor_t filterDesc,
                             hipdnnTensorFormat_t format,
                             hipdnnDataType_t dataType,
                             int k, int c, int h, int w);

hipdnnStatus_t
hipdnnSetFilterNdDescriptor( hipdnnFilterDescriptor_t filterDesc,
                             hipdnnDataType_t dataType,  /* image data type */
                             hipdnnTensorFormat_t format,
                             int nbDims,
                             const int filterDimA[]);

hipdnnStatus_t
hipdnnGetFilterNdDescriptor( const hipdnnFilterDescriptor_t filterDesc,
                             int nbDimsRequested,
                             hipdnnDataType_t *dataType,  /* image data type */
                             hipdnnTensorFormat_t *format,
                             int *nbDims,
                             int filterDimA[]);

hipdnnStatus_t
hipdnnGetFilter4dDescriptor( const hipdnnFilterDescriptor_t filterDesc,
                             hipdnnDataType_t *dataType,
                             hipdnnTensorFormat_t *format,
                             int *k, int *c, int *h, int *w);

hipdnnStatus_t
hipdnnDestroyFilterDescriptor( hipdnnFilterDescriptor_t filterDesc);

//========================== Convolutional =====================================

hipdnnStatus_t
hipdnnCreateConvolutionDescriptor( hipdnnConvolutionDescriptor_t *convDesc);

hipdnnStatus_t
hipdnnSetConvolution2dDescriptor( hipdnnConvolutionDescriptor_t convDesc,
                                  int pad_h, int pad_w, int u, int v,
                                  int upscalex, int upscaley,
                                  hipdnnConvolutionMode_t mode,
                                  hipdnnDataType_t computeType);

hipdnnStatus_t
hipdnnGetConvolution2dDescriptor( const hipdnnConvolutionDescriptor_t convDesc,
                                  int *pad_h, int *pad_y, int *u, int *v,
                                  int *upscalex, int *upscaley,
                                  hipdnnConvolutionMode_t *mode,
                                  hipdnnDataType_t *computeType);

hipdnnStatus_t
hipdnnGetConvolution2dForwardOutputDim(
                                const hipdnnConvolutionDescriptor_t convDesc,
                                const hipdnnTensorDescriptor_t inputTensorDesc,
                                const hipdnnFilterDescriptor_t filterDesc,
                                int *n, int *c, int *h, int *w);


hipdnnStatus_t
hipdnnSetConvolutionNdDescriptor( hipdnnConvolutionDescriptor_t convDesc,
                                  int arrayLength, /* nbDims-2 size */
                                  const int padA[],
                                  const int filterStrideA[],
                                  const int dilationA[],
                                  hipdnnConvolutionMode_t mode,
                                  hipdnnDataType_t computeType);  /* convolution data type */

hipdnnStatus_t
hipdnnDestroyConvolutionDescriptor( hipdnnConvolutionDescriptor_t convDesc);

hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithm( hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    const int requestedAlgoCount,
                                    int *returnedAlgoCount,
                                    hipdnnConvolutionFwdAlgoPerf_t *perfResults);

hipdnnStatus_t
hipdnnGetConvolutionForwardAlgorithm( hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    hipdnnConvolutionFwdPreference_t preference,
                                    size_t memoryLimitInBytes,
                                    hipdnnConvolutionFwdAlgo_t *algo);

hipdnnStatus_t
hipdnnFindConvolutionForwardAlgorithmEx( hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const void *w,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y,
                                    const int requestedAlgoCount,
                                    int *returnedAlgoCount,
                                    hipdnnConvolutionFwdAlgoPerf_t *perfResults,
                                    void *workSpace,
                                    size_t workSpaceSizeInBytes);

hipdnnStatus_t
hipdnnGetConvolutionForwardWorkspaceSize( hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    hipdnnConvolutionFwdAlgo_t algo,
                                    size_t *sizeInBytes);

hipdnnStatus_t
hipdnnConvolutionForward( hipdnnHandle_t handle,
                         const void *alpha,
                         const hipdnnTensorDescriptor_t xDesc,
                         const void *x,
                         const hipdnnFilterDescriptor_t wDesc,
                         const void *w,
                         const hipdnnConvolutionDescriptor_t convDesc,
                         hipdnnConvolutionFwdAlgo_t algo,
                         void *workSpace,
                         size_t workSpaceSizeInBytes,
                         const void *beta,
                         const hipdnnTensorDescriptor_t yDesc,
                         void *y);

hipdnnStatus_t
hipdnnConvolutionBackwardBias( hipdnnHandle_t handle,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const void *beta,
                                const hipdnnTensorDescriptor_t dbDesc,
                                void *db);

hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithm( hipdnnHandle_t handle,
                            const hipdnnTensorDescriptor_t xDesc,
                            const hipdnnTensorDescriptor_t dyDesc,
                            const hipdnnConvolutionDescriptor_t convDesc,
                            const hipdnnFilterDescriptor_t dwDesc,
                            const int requestedAlgoCount,
                            int *returnedAlgoCount,
                            hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults);

hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterAlgorithm( hipdnnHandle_t handle,
                              const hipdnnTensorDescriptor_t xDesc,
                              const hipdnnTensorDescriptor_t dyDesc,
                              const hipdnnConvolutionDescriptor_t convDesc,
                              const hipdnnFilterDescriptor_t dwDesc,
                              hipdnnConvolutionBwdFilterPreference_t preference,
                              size_t memoryLimitInBytes,
                              hipdnnConvolutionBwdFilterAlgo_t *algo);

hipdnnStatus_t
hipdnnFindConvolutionBackwardFilterAlgorithmEx( hipdnnHandle_t handle,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const void *dy,
                                const hipdnnConvolutionDescriptor_t convDesc,
                                const hipdnnFilterDescriptor_t dwDesc,
                                void *dw,
                                const int requestedAlgoCount,
                                int *returnedAlgoCount,
                                hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                void *workSpace,
                                size_t workSpaceSizeInBytes);

hipdnnStatus_t
hipdnnGetConvolutionBackwardFilterWorkspaceSize( hipdnnHandle_t handle,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const hipdnnTensorDescriptor_t dyDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnFilterDescriptor_t dwDesc,
                                    hipdnnConvolutionBwdFilterAlgo_t algo,
                                    size_t *sizeInBytes);

hipdnnStatus_t
hipdnnConvolutionBackwardFilter( hipdnnHandle_t handle,
                                  const void *alpha,
                                  const hipdnnTensorDescriptor_t xDesc,
                                  const void *x,
                                  const hipdnnTensorDescriptor_t dyDesc,
                                  const void *dy,
                                  const hipdnnConvolutionDescriptor_t convDesc,
                                  hipdnnConvolutionBwdFilterAlgo_t algo,
                                  void *workSpace,
                                  size_t workSpaceSizeInBytes,
                                  const void *beta,
                                  const hipdnnFilterDescriptor_t dwDesc,
                                  void *dw);

hipdnnStatus_t
hipdnnGetConvolutionBackwardDataWorkspaceSize( hipdnnHandle_t handle,
                                    const hipdnnFilterDescriptor_t wDesc,
                                    const hipdnnTensorDescriptor_t dyDesc,
                                    const hipdnnConvolutionDescriptor_t convDesc,
                                    const hipdnnTensorDescriptor_t dxDesc,
                                    hipdnnConvolutionBwdDataAlgo_t algo,
                                    size_t *sizeInBytes);

hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithm( hipdnnHandle_t handle,
                                const hipdnnFilterDescriptor_t wDesc,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const hipdnnConvolutionDescriptor_t convDesc,
                                const hipdnnTensorDescriptor_t dxDesc,
                                const int requestedAlgoCount,
                                int *returnedAlgoCount,
                                hipdnnConvolutionBwdDataAlgoPerf_t *perfResults);

hipdnnStatus_t
hipdnnGetConvolutionBackwardDataAlgorithm( hipdnnHandle_t handle,
                                const hipdnnFilterDescriptor_t wDesc,
                                const hipdnnTensorDescriptor_t dyDesc,
                                const hipdnnConvolutionDescriptor_t convDesc,
                                const hipdnnTensorDescriptor_t dxDesc,
                                hipdnnConvolutionBwdDataPreference_t preference,
                                size_t memoryLimitInBytes,
                                hipdnnConvolutionBwdDataAlgo_t *algo);

hipdnnStatus_t
hipdnnFindConvolutionBackwardDataAlgorithmEx( hipdnnHandle_t handle,
                               const hipdnnFilterDescriptor_t wDesc,
                               const void *w,
                               const hipdnnTensorDescriptor_t dyDesc,
                               const void *dy,
                               const hipdnnConvolutionDescriptor_t convDesc,
                               const hipdnnTensorDescriptor_t dxDesc,
                               void *dx,
                               const int requestedAlgoCount,
                               int *returnedAlgoCount,
                               hipdnnConvolutionBwdDataAlgoPerf_t *perfResults,
                               void *workSpace,
                               size_t workSpaceSizeInBytes);
hipdnnStatus_t
hipdnnConvolutionBackwardData( hipdnnHandle_t handle,
                               const void *alpha,
                               const hipdnnFilterDescriptor_t wDesc,
                               const void *w,
                               const hipdnnTensorDescriptor_t dyDesc,
                               const void *dy,
                               const hipdnnConvolutionDescriptor_t convDesc,
                               hipdnnConvolutionBwdDataAlgo_t algo,
                               void *workSpace,
                               size_t workSpaceSizeInBytes,
                               const void *beta,
                               const hipdnnTensorDescriptor_t dxDesc,
                               void *dx);

hipdnnStatus_t hipdnnSetConvolutionMathType(
    hipdnnConvolutionDescriptor_t convDesc, hipdnnMathType_t mathType);

hipdnnStatus_t
hipdnnSetConvolutionGroupCount( hipdnnConvolutionDescriptor_t convDesc,
                                int groupCount );

//============================== SoftMax =======================================

hipdnnStatus_t hipdnnSoftmaxForward( hipdnnHandle_t handle,
                                     hipdnnSoftmaxAlgorithm_t algo,
                                     hipdnnSoftmaxMode_t mode,
                                     const void *alpha,
                                     const hipdnnTensorDescriptor_t xDesc,
                                     const void *x,
                                     const void *beta,
                                     const hipdnnTensorDescriptor_t yDesc,
                                     void *y);

 hipdnnStatus_t hipdnnSoftmaxBackward( hipdnnHandle_t handle,
                                       hipdnnSoftmaxAlgorithm_t algo,
                                       hipdnnSoftmaxMode_t mode,
                                       const void *alpha,
                                       const hipdnnTensorDescriptor_t yDesc,
                                       const void *y,
                                       const hipdnnTensorDescriptor_t dyDesc,
                                       const void *dy,
                                       const void *beta,
                                       const hipdnnTensorDescriptor_t dxDesc,
                                       void *dx);

//================================ Pooling =====================================

hipdnnStatus_t
hipdnnCreatePoolingDescriptor(hipdnnPoolingDescriptor_t *poolingDesc);

hipdnnStatus_t
hipdnnSetPooling2dDescriptor( hipdnnPoolingDescriptor_t poolingDesc,
                              hipdnnPoolingMode_t mode,
                              hipdnnNanPropagation_t maxpoolingNanOpt,
                              int windowHeight,
                              int windowWidth,
                              int verticalPadding,
                              int horizontalPadding,
                              int verticalStride,
                              int horizontalStride);

hipdnnStatus_t
hipdnnSetPoolingNdDescriptor( hipdnnPoolingDescriptor_t poolingDesc,
                              const hipdnnPoolingMode_t mode,
                              const hipdnnNanPropagation_t maxpoolingNanOpt,
                              int nbDims,
                              const int windowDimA[],
                              const int paddingA[],
                              const int strideA[]);
hipdnnStatus_t
hipdnnGetPooling2dDescriptor( const hipdnnPoolingDescriptor_t poolingDesc,
                               hipdnnPoolingMode_t *mode,
                               hipdnnNanPropagation_t *maxpoolingNanOpt,
                               int *windowHeight,    int *windowWidth,
                               int *verticalPadding, int *horizontalPadding,
                               int *verticalStride,  int *horizontalStride);

hipdnnStatus_t
hipdnnGetPooling2dForwardOutputDim( const hipdnnPoolingDescriptor_t poolingDesc,
                                const hipdnnTensorDescriptor_t inputTensorDesc,
                                int *n, int *c, int *h, int *w);

hipdnnStatus_t
hipdnnDestroyPoolingDescriptor( hipdnnPoolingDescriptor_t poolingDesc);

hipdnnStatus_t hipdnnPoolingForward( hipdnnHandle_t handle,
                                    const hipdnnPoolingDescriptor_t poolingDesc,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const void *beta,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y, bool do_backward);

hipdnnStatus_t hipdnnPoolingBackward( hipdnnHandle_t handle,
                                    const hipdnnPoolingDescriptor_t poolingDesc,
                                    const void *alpha,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    const void *y,
                                    const hipdnnTensorDescriptor_t dyDesc,
                                    const void *dy,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x,
                                    const void *beta,
                                    const hipdnnTensorDescriptor_t dxDesc,
                                    void *dx);

//============================ Activation ======================================

hipdnnStatus_t
hipdnnCreateActivationDescriptor(hipdnnActivationDescriptor_t *activationDesc);

/* cudnn uses only one coeff param - for clipping threashold or alpha coefficient.
 * MIOpen supports three - Alpha, Beta and Gamma
 * The first parameter reluCeilingOrAlpha is the common denominator.
 * unless MIOpen specific mode used, zeros can be passed for activBeta and activExp
 */
hipdnnStatus_t
hipdnnSetActivationDescriptor( hipdnnActivationDescriptor_t activationDesc,
                                hipdnnActivationMode_t mode,
                                hipdnnNanPropagation_t reluNanOpt,
                                double reluCeilingOrAlpha,
                                double activBeta,
                                double activExp);

hipdnnStatus_t
hipdnnGetActivationDescriptor( const hipdnnActivationDescriptor_t activationDesc,
                                hipdnnActivationMode_t *mode,
                                hipdnnNanPropagation_t *reluNanOpt,
                                double *reluCeilingOrAlpha,
                                double *activBeta,
                                double *activExp);

hipdnnStatus_t
hipdnnDestroyActivationDescriptor(hipdnnActivationDescriptor_t activationDesc);

hipdnnStatus_t
hipdnnActivationForward( hipdnnHandle_t handle,
                         hipdnnActivationDescriptor_t activationDesc,
                         const void *alpha,
                         const hipdnnTensorDescriptor_t xDesc,
                         const void *x,
                         const void *beta,
                         const hipdnnTensorDescriptor_t yDesc,
                         void *y);

hipdnnStatus_t
hipdnnActivationBackward( hipdnnHandle_t handle,
                           hipdnnActivationDescriptor_t activationDesc,
                           const void *alpha,
                           const hipdnnTensorDescriptor_t yDesc,
                           const void *y,
                           const hipdnnTensorDescriptor_t dyDesc,
                           const void *dy,
                           const hipdnnTensorDescriptor_t xDesc,
                           const void *x,
                           const void *beta,
                           const hipdnnTensorDescriptor_t dxDesc,
                           void *dx);

//=======================Local Responce Normalization ==========================

hipdnnStatus_t
hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc);

hipdnnStatus_t
hipdnnSetLRNDescriptor( hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t mode,
                        unsigned lrnN,
                        double lrnAlpha,
                        double lrnBeta,
                        double lrnK);

hipdnnStatus_t
hipdnnGetLRNDescriptor( hipdnnLRNDescriptor_t normDesc,
                        hipdnnLRNMode_t *mode,
                        unsigned *lrnN,
                        double *lrnAlpha,
                        double *lrnBeta,
                        double *lrnK);

hipdnnStatus_t
hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t lrnDesc);

hipdnnStatus_t
hipdnnLRNCrossChannelForward( hipdnnHandle_t handle,
                              hipdnnLRNDescriptor_t normDesc,
                              hipdnnLRNMode_t lrnMode,
                              const void *alpha,
                              const hipdnnTensorDescriptor_t xDesc,
                              const void *x,
                              const void *beta,
                              const hipdnnTensorDescriptor_t yDesc,
                              void *y,
                              bool do_backward);

hipdnnStatus_t
hipdnnLRNCrossChannelForwardEx( hipdnnHandle_t handle,
                                hipdnnLRNDescriptor_t normDesc,
                                hipdnnLRNMode_t lrnMode,
                                const void *alpha,
                                const hipdnnTensorDescriptor_t xDesc,
                                const void *x,
                                const void *beta,
                                const hipdnnTensorDescriptor_t yDesc,
                                void *y,
                                size_t workspacesize,
                                void *workspace,
                                bool do_backward);

hipdnnStatus_t
hipdnnLRNCrossChannelBackward( hipdnnHandle_t handle,
                               hipdnnLRNDescriptor_t normDesc,
                               hipdnnLRNMode_t lrnMode,
                               const void *alpha,
                               const hipdnnTensorDescriptor_t yDesc,
                               const void *y,
                               const hipdnnTensorDescriptor_t dyDesc,
                               const void *dy,
                               const hipdnnTensorDescriptor_t xDesc,
                               const void *x,
                               const void *beta,
                               const hipdnnTensorDescriptor_t dxDesc,
                               void *dx);

hipdnnStatus_t
hipdnnLRNCrossChannelBackwardEx( hipdnnHandle_t handle,
                                  hipdnnLRNDescriptor_t normDesc,
                                  hipdnnLRNMode_t lrnMode,
                                  const void *alpha,
                                  const hipdnnTensorDescriptor_t yDesc,
                                  const void *y,
                                  const hipdnnTensorDescriptor_t dyDesc,
                                  const void *dy,
                                  const hipdnnTensorDescriptor_t xDesc,
                                  const void *x, const void *beta,
                                  const hipdnnTensorDescriptor_t dxDesc,
                                  void *dx,
                                  size_t workspacesize,
                                  void *workspace);

//============================ Batch Normalization =============================

hipdnnStatus_t hipdnnDeriveBNTensorDescriptor(
                                         hipdnnTensorDescriptor_t derivedBnDesc,
                                         const hipdnnTensorDescriptor_t xDesc,
                                         hipdnnBatchNormMode_t mode);

hipdnnStatus_t
hipdnnBatchNormalizationForwardTraining( hipdnnHandle_t handle,
                          hipdnnBatchNormMode_t mode,
                          void *alpha, void *beta,
                          const hipdnnTensorDescriptor_t xDesc,
                          const void *x,
                          const hipdnnTensorDescriptor_t yDesc,
                          void *y,
                          const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                          void *bnScale, void *bnBias,
                          double exponentialAverageFactor,
                          void *resultRunningMean,
                          void *resultRunningVariance,
                          double epsilon,
                          void *resultSaveMean,
                          void *resultSaveInvVariance);

hipdnnStatus_t
hipdnnnBatchNormalizationForwardInference( hipdnnHandle_t handle,
                           hipdnnBatchNormMode_t mode,
                           void *alpha, void *beta,
                           const hipdnnTensorDescriptor_t xDesc,
                           const void *x,
                           const hipdnnTensorDescriptor_t yDesc,
                           void *y,
                           const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                           const void *bnScale, const void *bnBias,
                           const void *estimatedMean,
                           const void *estimatedVariance,
                           double epsilon);

hipdnnStatus_t
hipdnnBatchNormalizationBackward( hipdnnHandle_t handle,
                             hipdnnBatchNormMode_t mode,
                             const void *alphaDataDiff,
                             const void *betaDataDiff,
                             const void *alphaParamDiff,
                             const void *betaParamDiff,
                             const hipdnnTensorDescriptor_t xDesc,
                             const void *x,
                             const hipdnnTensorDescriptor_t dyDesc,
                             const void *dy,
                             const hipdnnTensorDescriptor_t dxDesc,
                             void *dx,
                             const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc,
                             const void *bnScale,
                             void *resultBnScaleDiff,
                             void *resultBnBiasDiff,
                             double epsilon,
                             const void *savedMean,
                             const void *savedInvVariance);

hipdnnStatus_t
hipdnnBatchNormalizationForwardInference( hipdnnHandle_t handle,
                          hipdnnBatchNormMode_t mode,
                          const void *alpha,  // alpha[0] = result blend factor
                          const void *beta,   // beta[0] = dest layer blend factor
                          const hipdnnTensorDescriptor_t xDesc,
                          const void *x,  // NxCxHxW
                          const hipdnnTensorDescriptor_t yDesc,
                          void *y,  // NxCxHxW
                          const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
                          const void *bnScale, const void *bnBias,
                          const void *estimatedMean,
                          const void *estimatedVariance,
                          double epsilon);

//======================== Drop out Layer ======================================

hipdnnStatus_t
hipdnnCreateDropoutDescriptor( hipdnnDropoutDescriptor_t *dropoutDesc);

hipdnnStatus_t
hipdnnDropoutGetStatesSize(hipdnnHandle_t handle, size_t *sizeInBytes);

hipdnnStatus_t
hipdnnSetDropoutDescriptor( hipdnnDropoutDescriptor_t dropoutDesc,
                            hipdnnHandle_t handle,
                            float dropout,
                            void *states,
                            size_t stateSizeInBytes,
                            unsigned long long seed);

hipdnnStatus_t
hipdnnDestroyDropoutDescriptor( hipdnnDropoutDescriptor_t dropoutDesc);

//======================= Recurrent Neural Net =================================

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t *rnnDesc);

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc);

// Expensive. Creates the plan for the specific settings.
hipdnnStatus_t hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                             const int minibatch,
                                             const hipdnnDataType_t dataType,
                                             hipdnnPersistentRNNPlan_t *plan);

// Attaches the plan to the descriptor.
hipdnnStatus_t hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                          hipdnnPersistentRNNPlan_t plan);

hipdnnStatus_t hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan);

/* DataType in the RNN descriptor is used to determine math precision
 * DataType in weight descriptors and input descriptors is used to describe storage
 */

hipdnnStatus_t
hipdnnSetRNNDescriptor_v6( hipdnnHandle_t handle,
            hipdnnRNNDescriptor_t rnnDesc,
            const int hiddenSize,
            const int numLayers,
            hipdnnDropoutDescriptor_t dropoutDesc, /*Between layers, not between recurrent steps*/
            hipdnnRNNInputMode_t inputMode,
            hipdnnDirectionMode_t direction,
            hipdnnRNNMode_t mode,
            hipdnnRNNAlgo_t algo,
            hipdnnDataType_t dataType);

hipdnnStatus_t
hipdnnSetRNNDescriptor( hipdnnHandle_t handle,
            hipdnnRNNDescriptor_t rnnDesc,
            int hiddenSize,
            int numLayers,
            hipdnnDropoutDescriptor_t dropoutDesc,/*Between layers, not between recurrent steps*/
            hipdnnRNNInputMode_t inputMode,
            hipdnnDirectionMode_t direction,
            hipdnnRNNMode_t mode,
            hipdnnRNNAlgo_t algo,
            hipdnnDataType_t dataType,
            hipdnnRNNBiasMode_t biasMode);

hipdnnStatus_t
hipdnnSetRNNDescriptor_v5( hipdnnRNNDescriptor_t rnnDesc,
            int hiddenSize,
            int numLayers,
            hipdnnDropoutDescriptor_t dropoutDesc, /* Between layers, not between recurrent steps. */
            hipdnnRNNInputMode_t inputMode,
            hipdnnDirectionMode_t direction,
            hipdnnRNNMode_t mode,
            hipdnnDataType_t dataType);

hipdnnStatus_t
hipdnnGetRNNDescriptor(hipdnnHandle_t handle,
           hipdnnRNNDescriptor_t rnnDesc,
           int* hiddenSize, int* numLayers,
           hipdnnDropoutDescriptor_t *dropoutDesc,
           hipdnnRNNInputMode_t *inputMode,
           hipdnnDirectionMode_t *direction,
           hipdnnRNNMode_t *mode,
           hipdnnRNNAlgo_t *algo,
           hipdnnDataType_t *dataType,
           hipdnnRNNBiasMode_t *biasMode);

hipdnnStatus_t
hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
            const hipdnnRNNDescriptor_t rnnDesc,
            const hipdnnTensorDescriptor_t xDesc,
            size_t *sizeInBytes,
            hipdnnDataType_t dataType);

hipdnnStatus_t
hipdnnGetRNNLayerParamSize(hipdnnHandle_t handle,
                hipdnnRNNDescriptor_t rnnDesc,
                const int layer,
                hipdnnTensorDescriptor_t xDesc,
                const int paramID,
                size_t *numBytes);

hipdnnStatus_t
hipdnnGetRNNLayerBiasSize(hipdnnHandle_t handle,
                    hipdnnRNNDescriptor_t rnnDesc,
                    const int layer,
                    const int biasID,
                    size_t *numBytes);

hipdnnStatus_t hipdnnGetRNNWorkspaceSize(hipdnnHandle_t handle,
                                         const hipdnnRNNDescriptor_t rnnDesc,
                                         const int seqLength,
                                         const hipdnnTensorDescriptor_t *xDesc,
                                         size_t *sizeInBytes);

hipdnnStatus_t
hipdnnGetRNNTrainingReserveSize( hipdnnHandle_t handle,
                                 const hipdnnRNNDescriptor_t rnnDesc,
                                 const int seqLength,
                                 const hipdnnTensorDescriptor_t *xDesc,
                                 size_t *sizeInBytes);

hipdnnStatus_t hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
                                      const hipdnnRNNDescriptor_t rnnDesc,
                                      const hipdnnTensorDescriptor_t xDesc,
                                      size_t *sizeInBytes,
                                      hipdnnDataType_t dataType);

hipdnnStatus_t
hipdnnGetRNNLinLayerMatrixParams( hipdnnHandle_t handle,
                                  const hipdnnRNNDescriptor_t rnnDesc,
                                  const int layer,
                                  const hipdnnTensorDescriptor_t xDesc,
                                  const hipdnnFilterDescriptor_t wDesc,
                                  const void *w,
                                  const int linLayerID,
                                  hipdnnFilterDescriptor_t linLayerMatDesc,
                                  void **linLayerMat);

hipdnnStatus_t
hipdnnGetRNNLinLayerBiasParams( hipdnnHandle_t handle,
                                const hipdnnRNNDescriptor_t rnnDesc,
                                const int layer,
                                const hipdnnTensorDescriptor_t xDesc,
                                const hipdnnFilterDescriptor_t wDesc,
                                const void *w,
                                const int linLayerID,
                                hipdnnFilterDescriptor_t linLayerBiasDesc,
                                void **linLayerBias);

hipdnnStatus_t
hipdnnGetRNNParamsDescriptor(hipdnnHandle_t handle,
                             hipdnnRNNDescriptor_t rnnDesc,
                             hipdnnTensorDescriptor_t xDesc,
                             hipdnnTensorDescriptor_t wDesc,
                             hipdnnDataType_t dtype);

hipdnnStatus_t
hipdnnGetRNNInputTensorSize(hipdnnHandle_t handle,
                            hipdnnRNNDescriptor_t rnnDesc,
                            const int seqLen,
                            hipdnnTensorDescriptor_t *xDesc,
                            size_t *numBytes);

hipdnnStatus_t
hipdnnGetRNNHiddenTensorSize(hipdnnHandle_t handle,
                             hipdnnRNNDescriptor_t rnnDesc,
                             const int seqLen,
                             hipdnnTensorDescriptor_t *xDesc,
                             size_t *numBytes);

hipdnnStatus_t
hipdnnSetRNNLayerParam(hipdnnHandle_t handle,
                       hipdnnRNNDescriptor_t rnnDesc,
                       const int layer,
                       hipdnnTensorDescriptor_t xDesc,
                       hipdnnTensorDescriptor_t wDesc,
                       void *w,
                       const int paramID,
                       hipdnnTensorDescriptor_t paramDesc,
                       const void *layerParam);

hipdnnStatus_t
hipdnnSetRNNLayerBias(hipdnnHandle_t handle,
                      hipdnnRNNDescriptor_t rnnDesc,
                      const int layer,
                      hipdnnTensorDescriptor_t xDesc,
                      hipdnnTensorDescriptor_t wDesc,
                      void *w,
                      const int biasID,
                      hipdnnTensorDescriptor_t biasDesc,
                      const void *layerBias);

hipdnnStatus_t
hipdnnRNNForwardInference( hipdnnHandle_t handle,
                           const hipdnnRNNDescriptor_t rnnDesc,
                           const int seqLength,
                           const hipdnnTensorDescriptor_t *xDesc, const void *x,
                           const hipdnnTensorDescriptor_t hxDesc, const void *hx,
                           const hipdnnTensorDescriptor_t cxDesc, const void *cx,
                           const hipdnnFilterDescriptor_t wDesc,  const void *w,
                           const hipdnnTensorDescriptor_t *yDesc, void *y,
                           const hipdnnTensorDescriptor_t hyDesc, void *hy,
                           const hipdnnTensorDescriptor_t cyDesc, void *cy,
                           void *workspace,
                           size_t workSpaceSizeInBytes);

hipdnnStatus_t
hipdnnRNNForwardTraining( hipdnnHandle_t handle,
                          const hipdnnRNNDescriptor_t rnnDesc,
                          const int seqLength,
                          const hipdnnTensorDescriptor_t *xDesc, const void *x,
                          const hipdnnTensorDescriptor_t hxDesc, const void *hx,
                          const hipdnnTensorDescriptor_t cxDesc, const void *cx,
                          const hipdnnFilterDescriptor_t wDesc,  const void *w,
                          const hipdnnTensorDescriptor_t *yDesc, void *y,
                          const hipdnnTensorDescriptor_t hyDesc, void *hy,
                          const hipdnnTensorDescriptor_t cyDesc, void *cy,
                          void *workspace, size_t workSpaceSizeInBytes,
                          void *reserveSpace, size_t reserveSpaceSizeInBytes);

hipdnnStatus_t
hipdnnRNNBackwardData( hipdnnHandle_t handle,
                       const hipdnnRNNDescriptor_t rnnDesc,
                       const int seqLength,
                       const hipdnnTensorDescriptor_t *yDesc,  const void *y,
                       const hipdnnTensorDescriptor_t *dyDesc, const void *dy,
                       const hipdnnTensorDescriptor_t dhyDesc, const void *dhy,
                       const hipdnnTensorDescriptor_t dcyDesc, const void *dcy,
                       const hipdnnFilterDescriptor_t wDesc,   const void *w,
                       const hipdnnTensorDescriptor_t hxDesc,  const void *hx,
                       const hipdnnTensorDescriptor_t cxDesc,  const void *cx,
                       const hipdnnTensorDescriptor_t *dxDesc, void *dx,
                       const hipdnnTensorDescriptor_t dhxDesc, void *dhx,
                       const hipdnnTensorDescriptor_t dcxDesc, void *dcx,
                       void *workspace, size_t workSpaceSizeInBytes,
                       void *reserveSpace, size_t reserveSpaceSizeInBytes);

hipdnnStatus_t
hipdnnRNNBackwardWeights( hipdnnHandle_t handle,
                          const hipdnnRNNDescriptor_t rnnDesc,
                          const int seqLength,
                          const hipdnnTensorDescriptor_t *xDesc, const void *x,
                          const hipdnnTensorDescriptor_t hxDesc, const void *hx,
                          const hipdnnTensorDescriptor_t *yDesc, const void *y,
                          const void *workspace, size_t workSpaceSizeInBytes,
                          const hipdnnFilterDescriptor_t dwDesc, void *dw,
                          const void *reserveSpace, size_t reserveSpaceSizeInBytes);

//========================== Fusion API ========================================

hipdnnStatus_t
hipdnnCreateFusionPlan( hipdnnFusionPlanDescriptor_t *fusePlanDesc,
                        const hipdnnFusionDirection_t fuseDirection,
                        const hipdnnTensorDescriptor_t inputDesc);


hipdnnStatus_t
hipdnnCreateOpConvForward( hipdnnFusionPlanDescriptor_t    fusePlanDesc,
                           hipdnnFusionOpDescriptor_t*     convOp,
                           hipdnnConvolutionDescriptor_t   convDesc,
                           const hipdnnTensorDescriptor_t  wDesc );

hipdnnStatus_t
hipdnnCreateOpBiasForward( hipdnnFusionPlanDescriptor_t fusePlanDesc,
                           hipdnnFusionOpDescriptor_t *biasOp,
                           const hipdnnTensorDescriptor_t bDesc);

hipdnnStatus_t
hipdnnCreateOpActivationForward( hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                 hipdnnFusionOpDescriptor_t *activOp,
                                 hipdnnActivationMode_t mode);

hipdnnStatus_t
hipdnnCreateOpBatchNormInference(
                 hipdnnFusionPlanDescriptor_t fusePlanDesc,
                 hipdnnFusionOpDescriptor_t *bnOp,
                 const hipdnnBatchNormMode_t bn_mode,
                 const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc);

hipdnnStatus_t
hipdnnCompileFusionPlan( hipdnnHandle_t handle,
                         hipdnnFusionPlanDescriptor_t fusePlanDesc);

hipdnnStatus_t
hipdnnFusionPlanGetOp( hipdnnFusionPlanDescriptor_t fusePlanDesc,
                       const int op_idx,
                       hipdnnFusionOpDescriptor_t *op);

hipdnnStatus_t
hipdnnFusionPlanGetWorkSpaceSize( hipdnnHandle_t handle,
                                  hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                  size_t *workSpaceSize,
                                  hipdnnConvolutionFwdAlgo_t algo);

hipdnnStatus_t
hipdnnFusionPlanConvolutionGetAlgo(
                        hipdnnFusionPlanDescriptor_t fusePlanDesc,
                        const int requestAlgoCount,
                        int* returnedAlgoCount,
                        hipdnnConvolutionFwdAlgo_t* returnedAlgos);

hipdnnStatus_t hipdnnCreateOperatorArgs( hipdnnOperatorArgs_t* args);

hipdnnStatus_t
hipdnnSetOpArgsConvForward( hipdnnOperatorArgs_t args,
                            const hipdnnFusionOpDescriptor_t convOp,
                            const void *alpha,
                            const void *beta,
                            const void *w );

hipdnnStatus_t
hipdnnSetOpArgsBiasForward( hipdnnOperatorArgs_t args,
                            const hipdnnFusionOpDescriptor_t biasOp,
                            const void *alpha,
                            const void *beta,
                            const void *bias );

hipdnnStatus_t
hipdnnSetOpArgsActivForward( hipdnnOperatorArgs_t args,
                             const hipdnnFusionOpDescriptor_t activOp,
                             const void *alpha,
                             const void *beta,
                             double activAlpha,
                             double activBeta,
                             double activGamma);

hipdnnStatus_t
hipdnnSetOpArgsBatchNormInference( hipdnnOperatorArgs_t args,
                                   const hipdnnFusionOpDescriptor_t bnOp,
                                   const void* alpha,
                                   const void* beta,
                                   const void* bnScale,
                                   const void* bnBias,
                                   const void* estimatedMean,
                                   const void* estimatedVariance,
                                   double epsilon);


hipdnnStatus_t
hipdnnExecuteFusionPlan( const hipdnnHandle_t handle,
                         const hipdnnFusionPlanDescriptor_t fusePlanDesc,
                         const hipdnnTensorDescriptor_t inputDesc,
                         const void *input,
                         const hipdnnTensorDescriptor_t outputDesc, void *output,
                         hipdnnOperatorArgs_t args);

hipdnnStatus_t hipdnnDestroyOperatorArgs( hipdnnOperatorArgs_t args);

hipdnnStatus_t
hipdnnDestroyFusionPlan( hipdnnFusionPlanDescriptor_t fusePlanDesc);

//==============================================================================

const char *hipdnnGetErrorString(hipdnnStatus_t status);


#ifdef __cplusplus
}
#endif

#endif //WITH_ROCM

#endif  // HIPDNN_H
