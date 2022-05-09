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
#ifdef WITH_HIP

#include <assert.h>
#include <hipdnn_miopen.h>
#include <hipdnn.h>
#include <logger.h>
#include <stdint.h>
#include <exception>
#include <iterator>
#include <map>
#include "hip/hip_runtime.h"

#define CHECK_MIO(expression)                                                   \
    {                                                                           \
        hipdnnStatus_t status = miopenTohipdnnStatus(expression);               \
        if (status != HIPDNN_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "HIPDNN error: %s (%d) at %s:%d\n",                 \
                    hipdnnGetErrorString(status), status, __FILE__, __LINE__);  \
            return status;                                                      \
        }                                                                       \
    }

#define CHECK_HIPDNN_NO_RET(expression)                                         \
    {                                                                           \
        hipdnnStatus_t error = (expression);                                    \
        if (error != HIPDNN_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "HIPDNN error: '%s'(%d) at %s:%d\n",                \
                    hipdnnGetErrorString(error), error, __FILE__, __LINE__);    \
        }                                                                       \
    }

#define HIPDNNFLUSH << std::flush;
#define PROMOTE_TO_SUPPORTED

#ifndef thread_local
#if __STDC_VERSION__ >= 201112 && !defined __STDC_NO_THREADS__
#define thread_local _Thread_local
#elif defined _WIN32 && (defined _MSC_VER || defined __ICL || \
                         defined __DMC__ || defined __BORLANDC__)
#define thread_local __declspec(thread)
/* note that ICC (linux) and Clang are covered by __GNUC__ */
#elif defined __GNUC__ || defined __SUNPRO_C || defined __xlC__
#define thread_local __thread
#else
//#error "Cannot define thread_local"
// NOT THREAD SAFE.
#define thread_local
#endif
#endif

// std::map<miopenPoolingDescriptor_t, void *>  sPoolingDescToWorkspace;  ????

static std::map<miopenTensorDescriptor_t, std::pair<int8_t *, size_t>>
    sDescToWorkspacePooling;  // device pointers

static std::map<miopenTensorDescriptor_t, std::pair<int8_t *, size_t>>
    sDescToWorkspaceLRN;  // device pointers

static std::map<miopenConvolutionDescriptor_t, int *>
    sDescTo3DConvolution;  // To bookkeep 3D depth information


// Custom TensorAdd Kernel

/*
 * dst= dst + beta * prior
 */
template <typename T>
__global__ void TensorAdd(T *C_d, T *A_d, T beta, int N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;
    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = beta * A_d[i] + C_d[i];
    }
}

// SaveAsPriorBuffer routine to book keep and return priorData before any
// activation or convolution

// Returns the hipMalloc'ed PriorData to be used for accumalation when
// the scaling factor beta is non zero
void *SaveAsPriorBuffer(void *dData) {
    void *dPrior = NULL;    // Pointer to keep track of priorDst value
    size_t dPriorSize = 0;  // PriorDstSize
    CHECK_HIP(hipMemPtrGetInfo(
        dData, &dPriorSize));  // Get the info of the gradient dx size
    CHECK_HIP(hipMalloc(&dPrior, dPriorSize));  // Allocate priorDst
    CHECK_HIP(hipMemcpy(
        dPrior, dData, dPriorSize,
        hipMemcpyDeviceToDevice));  // Copy gradient to prior Destination
    return dPrior;
}

// Revoke the PriorBuffer
void deallocPrior(void *dData) {
    size_t dPriorSize = 0;  // PriorDstSize
    CHECK_HIP(hipMemPtrGetInfo(dData, &dPriorSize));
    if (dPriorSize > 0) CHECK_HIP(hipFree(dData));
}

//=============================================================================

hipdnnStatus_t miopenTohipdnnStatus(miopenStatus_t cStatus) {
    switch (cStatus) {
        case miopenStatusSuccess:
            return HIPDNN_STATUS_SUCCESS;
            break;
        case miopenStatusNotInitialized:
            return HIPDNN_STATUS_NOT_INITIALIZED;
            break;
        case miopenStatusAllocFailed:
            return HIPDNN_STATUS_ALLOC_FAILED;
            break;
        case miopenStatusBadParm:
            return HIPDNN_STATUS_BAD_PARAM;
            break;
        case miopenStatusInternalError:
            return HIPDNN_STATUS_INTERNAL_ERROR;
            break;
        case miopenStatusInvalidValue:
            return HIPDNN_STATUS_INVALID_VALUE;
            break;
        case miopenStatusUnknownError:
            return HIPDNN_STATUS_EXECUTION_FAILED;
            break;
        case miopenStatusNotImplemented:
            return HIPDNN_STATUS_NOT_SUPPORTED;
            break;
        default:
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipTomiopenDataType(hipdnnDataType_t in, miopenDataType_t *out) {
    switch (in) {
        case HIPDNN_DATA_FLOAT:
            *out = miopenFloat;
            break;
        case HIPDNN_DATA_HALF:
            *out = miopenHalf;
            break;
        case HIPDNN_DATA_DOUBLE:
        case HIPDNN_DATA_INT8:
        case HIPDNN_DATA_INT32:
        case HIPDNN_DATA_INT8x4:
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenDataType " << in << ": NOT SUPPORTED."
                                                     << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipDataType(miopenDataType_t in, hipdnnDataType_t *out) {
    switch (in) {
        case miopenFloat:
            *out = HIPDNN_DATA_FLOAT;
            break;
        case miopenHalf:
            *out = HIPDNN_DATA_HALF;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipDataType " << in << ": NOT SUPPORTED."
                                                     << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t miopenTohipdnnConvolutionMode(miopenConvolutionMode_t in,
                                                hipdnnConvolutionMode_t* out) {
    if (in == miopenConvolution) //MIOpen's convolution is cudnn's corss corelation equivalent
        *out = HIPDNN_CROSS_CORRELATION;
    else if( in == miopenTranspose)
        *out = HIPDNN_TRANSPOSE;
    else if( in == miopenGroupConv)
        *out = HIPDNN_GROUP_CONVOLUTION;
    else if( in == miopenDepthwise)
        *out = HIPDNN_DEPTHWISE;
    else
        return HIPDNN_STATUS_NOT_SUPPORTED;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipTomiopenConvolutionMode(hipdnnConvolutionMode_t in,
                                                miopenConvolutionMode_t* out) {
    if (in == HIPDNN_CROSS_CORRELATION)
        *out = miopenConvolution;
    else if( in == HIPDNN_TRANSPOSE)
        *out = miopenTranspose;
    else if( in == HIPDNN_GROUP_CONVOLUTION)
        *out = miopenGroupConv;
    else if( in == HIPDNN_DEPTHWISE)
        *out = miopenDepthwise;

    else if( in == HIPDNN_CONVOLUTION) {
        std::cerr << "CONVOLUTION is Not supported in MIOpen."
                  <<"CROSS_CORRELATION can be used instead for (Train+Inference)";
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    else
        return HIPDNN_STATUS_NOT_SUPPORTED;

    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipTomiopenPoolingMode(hipdnnPoolingMode_t in,
                                      miopenPoolingMode_t *out) {
    switch (in) {
        case HIPDNN_POOLING_MAX:
            *out = miopenPoolingMax;
            break;
        case HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING:
            *out = miopenPoolingAverageInclusive;
            break;
        case HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING:
            *out = miopenPoolingAverage;
            break;
        case HIPDNN_POOLING_MAX_DETERMINISTIC:
            *out = miopenPoolingMax;
            break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenPoolingMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipPoolingMode(miopenPoolingMode_t in,
                                      hipdnnPoolingMode_t *out) {
    switch (in) {
        case miopenPoolingMax:
            *out = HIPDNN_POOLING_MAX;
            break;
        case miopenPoolingAverageInclusive:
            *out = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case miopenPoolingAverage:
            *out = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
            // HGSOS     *out = HIPDNN_POOLING_MAX_DETERMINISTIC;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipPoolingMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipTomiopenLRNMode(hipdnnLRNMode_t in, miopenLRNMode_t *out) {
    switch (in) {
        case HIPDNN_LRN_WITHIN_CHANNEL:
            *out = miopenLRNWithinChannel;
            break;
        case HIPDNN_LRN_CROSS_CHANNEL:
            *out = miopenLRNCrossChannel;
            break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenLRNMode" << in << ": NOT SUPPORTED."
                                                   << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipLRNMode(miopenLRNMode_t in, hipdnnLRNMode_t *out) {
    switch (in) {
        case miopenLRNWithinChannel:
            *out = HIPDNN_LRN_WITHIN_CHANNEL;
            break;
        case miopenLRNCrossChannel:
            *out = HIPDNN_LRN_CROSS_CHANNEL;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipLRNMode " << in << ": NOT SUPPORTED."
                                                    << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipTomiopenBatchNormMode(hipdnnBatchNormMode_t in,
                                        miopenBatchNormMode_t *out) {
    switch (in) {
        case HIPDNN_BATCHNORM_PER_ACTIVATION:
            *out = miopenBNPerActivation;
            break;
        case HIPDNN_BATCHNORM_SPATIAL:
            *out = miopenBNSpatial;
            break;
        case HIPDNN_BATCHNORM_SPATIAL_PERSISTENT:
            *out = miopenBNSpatial;  // TODO: Change when Spatial persistent is
                                     // supported on MIOPEN
            break;
        default:
            HIPDNN_OPEN_LOG_E("Invalid HIPDNN_BATCHNORM_MODE" << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t miopenTohipActivationMode(miopenActivationMode_t in,
                                         hipdnnActivationMode_t *out) {
    switch (in) {
        case miopenActivationLOGISTIC:
            *out = HIPDNN_ACTIVATION_SIGMOID;
            break;

        case miopenActivationRELU:
            *out = HIPDNN_ACTIVATION_RELU;
            break;

        case miopenActivationTANH:
            *out = HIPDNN_ACTIVATION_TANH;
            break;

        case miopenActivationPASTHRU:
            *out = HIPDNN_ACTIVATION_PATHTRU;
            break;

        case miopenActivationSOFTRELU:
            *out = HIPDNN_ACTIVATION_SOFTRELU;
            break;

        case miopenActivationABS:
            *out = HIPDNN_ACTIVATION_ABS;
            break;

        case miopenActivationPOWER:
            *out = HIPDNN_ACTIVATION_POWER;
            break;

        default:
            HIPDNN_OPEN_LOG_M("miopenTohipActivationMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipTomiopenActivationMode(hipdnnActivationMode_t in,
                                         miopenActivationMode_t *out) {
    switch (in) {
        case HIPDNN_ACTIVATION_SIGMOID:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_SIGMOID" << std::flush);
            *out = miopenActivationLOGISTIC;
            break;

        case HIPDNN_ACTIVATION_RELU:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_RELU" << std::flush);
            *out = miopenActivationRELU;
            break;

        case HIPDNN_ACTIVATION_TANH:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_TANH" << std::flush);
            *out = miopenActivationTANH;
            break;

        case HIPDNN_ACTIVATION_PATHTRU:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_PATHTRU" << std::flush);
            *out = miopenActivationPASTHRU;
            break;

        case HIPDNN_ACTIVATION_SOFTRELU:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_SOFTRELU" << std::flush);
            *out = miopenActivationSOFTRELU;
            break;

        case HIPDNN_ACTIVATION_ABS:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_ABS" << std::flush);
            *out = miopenActivationABS;
            break;

        case HIPDNN_ACTIVATION_POWER:
            HIPDNN_OPEN_LOG_M("HIPDNN_ACTIVATION_POWER" << std::flush);
            *out = miopenActivationPOWER;
            break;

        case HIPDNN_ACTIVATION_ELU:
            HIPDNN_OPEN_LOG_E("HIPDNN_ACTIVATION_ELU" << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;

        case HIPDNN_ACTIVATION_CLIPPED_RELU:
            HIPDNN_OPEN_LOG_E("HIPDNN_ACTIVATION_CLIPPED_RELU" << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;

        default:
            HIPDNN_OPEN_LOG_M("miopenTohipPoolingMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionFwdAlgo(hipdnnConvolutionFwdAlgo_t in,
                                             miopenConvFwdAlgorithm_t *out) {
    switch (in) {
        case HIPDNN_CONVOLUTION_FWD_ALGO_GEMM:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_GEMM" << std::flush);
            *out = miopenConvolutionFwdAlgoGEMM;
            break;

        case HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT"
                              << std::flush);
            *out = miopenConvolutionFwdAlgoDirect;
            break;

        case HIPDNN_CONVOLUTION_FWD_ALGO_FFT:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_FFT" << std::flush);
            *out = miopenConvolutionFwdAlgoFFT;
            break;

        case HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"
                              << std::flush);
            *out = miopenConvolutionFwdAlgoWinograd;
            break;

        case HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
                              << std::flush);
            *out = miopenConvolutionFwdAlgoGEMM;
            break;

        default:
            HIPDNN_OPEN_LOG_E("hipdnnConvolutionFwdAlgo_t: "
                              << in << " NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipConvolutionFwdAlgo(miopenConvFwdAlgorithm_t in,
                                             hipdnnConvolutionFwdAlgo_t *out) {
    switch (in) {
        case miopenConvolutionFwdAlgoGEMM:
            *out = HIPDNN_CONVOLUTION_FWD_ALGO_GEMM;
            break;
        case miopenConvolutionFwdAlgoDirect:
            *out = HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT;
            break;
        case miopenConvolutionFwdAlgoFFT:
            *out = HIPDNN_CONVOLUTION_FWD_ALGO_FFT;
            break;
        case miopenConvolutionFwdAlgoWinograd:
            *out = HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipConvolutionFwdAlgo "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
            return HIPDNN_STATUS_NOT_SUPPORTED;
            break;
    }
    return HIPDNN_STATUS_SUCCESS;
}

int ConvolutionFwdAlgoCount() { return 4; }

// call ConvolutionFwdAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionFwdAlgo_t GetConvolutionFwdAlgo(int i) {
    hipdnnConvolutionFwdAlgo_t retVal;
    miopenConvFwdAlgorithm_t mialgo;

    if (i < ConvolutionFwdAlgoCount()) {
        mialgo = (miopenConvFwdAlgorithm_t)i;
    } else {
        // for protection
        mialgo = (miopenConvFwdAlgorithm_t)miopenConvolutionFwdAlgoWinograd;
    }
    CHECK_HIPDNN_NO_RET( miopenTohipConvolutionFwdAlgo(mialgo, &retVal));
    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdFilterAlgo(
    hipdnnConvolutionBwdFilterAlgo_t in, miopenConvBwdWeightsAlgorithm_t *out) {
    switch (in) {
        case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0"
                              << std::flush);
            *out = miopenConvolutionBwdWeightsAlgoGEMM;
            break;

        case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1"
                              << std::flush);
            *out = miopenConvolutionBwdWeightsAlgoDirect;
            break;

        case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD"
                              << std::flush);
            *out = miopenConvolutionBwdWeightsAlgoWinograd;
            break;

            // TODO NEEL: Add other BwdFilter algorithms
            /*case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
             case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
             case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
             case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
             case HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:*/ //TODO: will be added in future
        default:
            HIPDNN_OPEN_LOG_E("hipdnnConvolutionBwdFilterAlgo_t: "
                              << in << " NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
            break;
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipConvolutionBwdFilterAlgo(
    miopenConvBwdWeightsAlgorithm_t in, hipdnnConvolutionBwdFilterAlgo_t *out) {
    switch (in) {
        case miopenConvolutionBwdWeightsAlgoGEMM:
            *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
            break;
        case miopenConvolutionBwdWeightsAlgoDirect:
            *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            break;
        case miopenConvolutionBwdWeightsAlgoWinograd:
            *out = HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
            break;
        default:
            HIPDNN_OPEN_LOG_E("miopenTohipConvolutionBwdFilterAlgo: "
                              << in << " NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

int ConvolutionBwdFilterAlgoCount() { return (int)2; }

// call ConvolutionBwdFilterAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdFilterAlgo_t GetConvolutionBwdFilterAlgo(int i) {
    hipdnnConvolutionBwdFilterAlgo_t retVal;
    miopenConvBwdWeightsAlgorithm_t mialgo;

    if (i < ConvolutionBwdFilterAlgoCount()) {
        mialgo = (miopenConvBwdWeightsAlgorithm_t)i;
    } else {
        // for protection
        mialgo = (miopenConvBwdWeightsAlgorithm_t)
            miopenConvolutionBwdWeightsAlgoGEMM;
    }
    CHECK_HIPDNN_NO_RET(miopenTohipConvolutionBwdFilterAlgo(mialgo, &retVal));

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipTomiopenConvolutionBwdDataAlgo(
    hipdnnConvolutionBwdDataAlgo_t in, miopenConvBwdDataAlgorithm_t *out) {
    switch (in) {
        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0"
                              << std::flush);
            *out = miopenConvolutionBwdDataAlgoGEMM;
            break;

        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1"
                              << std::flush);
            *out = miopenConvolutionBwdDataAlgoDirect;
            break;

        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD"
                              << std::flush);
            *out = miopenConvolutionBwdDataAlgoWinograd;
            break;

        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            HIPDNN_OPEN_LOG_M(
                "HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED"
                << std::flush);
            *out = miopenConvolutionBwdDataAlgoWinograd;
            break;

        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT"
                              << std::flush);
            *out = miopenConvolutionBwdDataAlgoFFT;
            break;

        case HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM:
            HIPDNN_OPEN_LOG_M("HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM"
                              << std::flush);
            *out = miopenTransposeBwdDataAlgoGEMM;
            break;

        default:
            HIPDNN_OPEN_LOG_E("hipdnnConvolutionBwdDataAlgo_t: "
                              << in << " NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipConvolutionBwdDataAlgo(
    miopenConvBwdDataAlgorithm_t in, hipdnnConvolutionBwdDataAlgo_t *out) {
    switch (in) {
        case miopenConvolutionBwdDataAlgoGEMM:
            *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0;
            break;
        case miopenConvolutionBwdDataAlgoDirect:
            *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            break;
        case miopenConvolutionBwdDataAlgoFFT:
            *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
            break;
        case miopenConvolutionBwdDataAlgoWinograd:
            *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
            break;
        case miopenTransposeBwdDataAlgoGEMM:
            *out = HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM;
            break;
        default:
            HIPDNN_OPEN_LOG_E("miopenTohipConvolutionBwdDataAlgo: "
                              << in << " NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

int ConvolutionBwdDataAlgoCount() { return (int)2; }

// call ConvolutionBwdDataAlgoCount first, caller's responsibility to
// make sure that i is not too large!
//
hipdnnConvolutionBwdDataAlgo_t GetConvolutionBwdDataAlgo(int i) {
    hipdnnConvolutionBwdDataAlgo_t retVal;
    miopenConvBwdDataAlgorithm_t mialgo;

    if (i < ConvolutionBwdDataAlgoCount()) {
        mialgo = (miopenConvBwdDataAlgorithm_t)i;
    } else {
        // for protection
        mialgo =
            (miopenConvBwdDataAlgorithm_t)miopenConvolutionBwdDataAlgoWinograd;
    }
    CHECK_HIPDNN_NO_RET(miopenTohipConvolutionBwdDataAlgo(mialgo, &retVal));

    return retVal;
}

//=============================================================================

hipdnnStatus_t hipSoftmaxModeSupported(hipdnnSoftmaxMode_t in) {
    switch (in) {
        // PRNSOS: MAX mode need to check
        case HIPDNN_SOFTMAX_MODE_INSTANCE:
            HIPDNN_OPEN_LOG_E("HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED."
                              << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;

        case HIPDNN_SOFTMAX_MODE_CHANNEL:
            break;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t SoftmaxAlgorithmSupported(hipdnnSoftmaxAlgorithm_t in) {
    switch (in) {
        case HIPDNN_SOFTMAX_FAST:
        case HIPDNN_SOFTMAX_ACCURATE:
            break;
        case HIPDNN_SOFTMAX_LOG:
            return HIPDNN_STATUS_NOT_SUPPORTED;
            break;
    }
    return HIPDNN_STATUS_SUCCESS;
}

// miopen does not define tensor format,
// implicitly HIPDNN_TENSOR_NCHW only
hipdnnStatus_t hipTensorFormatSupported(hipdnnTensorFormat_t in) {
    if (in == HIPDNN_TENSOR_NCHW) {
        HIPDNN_OPEN_LOG_M("HIPDNN_TENSOR_NCHW" << std::flush);
        return HIPDNN_STATUS_SUCCESS;
    } else {
        HIPDNN_OPEN_LOG_E("hipdnnTensorFormat_t " << in << " NOT SUPPORTED."
                                                  << std::flush);
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
}

hipdnnStatus_t ConvolutionFwdPreferenceSupported(
    hipdnnConvolutionFwdPreference_t in) {
    switch (in) {
        case HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE:
            return HIPDNN_STATUS_NOT_SUPPORTED;

        case HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST:
            break;
        case HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT:
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t ConvolutionBwdFilterPreferenceSupported(
    hipdnnConvolutionBwdFilterPreference_t in) {
    switch (in) {
        case HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE:
            return HIPDNN_STATUS_NOT_SUPPORTED;
        case HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST:
            break;
        case HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT:
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//==================================RNN Operations =============================

hipdnnStatus_t hipTomiopenRNNInputMode(hipdnnRNNInputMode_t in,
                                      miopenRNNInputMode_t *out) {
    switch (in) {
        case HIPDNN_LINEAR_INPUT:
            *out = miopenRNNlinear;
            break;
        case HIPDNN_SKIP_INPUT:
            *out = miopenRNNskip;
            break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenRNNInputMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipRNNInputMode(miopenRNNInputMode_t in,
                                      hipdnnRNNInputMode_t *out) {
    switch (in) {
        case miopenRNNlinear:
            *out = HIPDNN_LINEAR_INPUT;
            break;
        case miopenRNNskip:
            *out = HIPDNN_SKIP_INPUT;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipRNNInputMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//====================================================================================

hipdnnStatus_t hipTomiopenRNNDirectionMode(hipdnnDirectionMode_t in,
                                       miopenRNNDirectionMode_t *out) {
    switch (in) {
        case HIPDNN_UNIDIRECTIONAL:
            *out = miopenRNNunidirection;
            break;
        case HIPDNN_BIDIRECTIONAL:
            *out = miopenRNNbidirection;
            break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenRNNDirectionMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipRNNDirectionMode(miopenRNNDirectionMode_t in,
                                        hipdnnDirectionMode_t *out) {
    switch (in) {
        case miopenRNNunidirection:
            *out = HIPDNN_UNIDIRECTIONAL ;
            break;
        case miopenRNNbidirection:
            *out = HIPDNN_BIDIRECTIONAL;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipRNNDirectionMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=====================================================================================

hipdnnStatus_t hipTomiopenRNNMode(hipdnnRNNMode_t in,
                                miopenRNNMode_t *out) {
    switch (in) {
        case HIPDNN_RNN_RELU:
            *out = miopenRNNRELU;
            break;
        case HIPDNN_RNN_TANH:
            *out = miopenRNNTANH;
            break;
        case HIPDNN_LSTM:
            *out = miopenLSTM;
            break;
        case HIPDNN_GRU:
            *out = miopenGRU;
            break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenRNNMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipRNNMode(miopenRNNMode_t in,
                                hipdnnRNNMode_t *out) {
    switch (in) {
        case miopenRNNRELU:
            *out = HIPDNN_RNN_RELU ;
            break;
        case miopenRNNTANH:
            *out = HIPDNN_RNN_TANH ;
            break;
        case miopenLSTM:
            *out = HIPDNN_LSTM ;
            break;
        case miopenGRU:
            *out = HIPDNN_GRU ;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipRNNMode "
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//==========================================================================================

hipdnnStatus_t hipTomiopenRNNAlgo(hipdnnRNNAlgo_t in,
                                        miopenRNNAlgo_t *out) {
    switch (in) {
        case HIPDNN_RNN_ALGO_STANDARD:
            *out = miopenRNNdefault;
            break;
        case HIPDNN_RNN_ALGO_PERSIST_STATIC:
        case HIPDNN_RNN_ALGO_PERSIST_DYNAMIC:
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenRNNAlgo"
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipRNNAlgo(miopenRNNAlgo_t in,
                                hipdnnRNNAlgo_t *out) {
    switch (in) {
        case miopenRNNdefault:
            *out = HIPDNN_RNN_ALGO_STANDARD;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipRNNAlgo"
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//============================================================================================================

hipdnnStatus_t hipTomiopenRNNBias(hipdnnRNNBiasMode_t in,
                                        miopenRNNBiasMode_t *out) {

    switch (in) {
        case HIPDNN_RNN_NO_BIAS:
            *out = miopenRNNNoBias;
            break;
        case HIPDNN_RNN_WITH_BIAS:
            *out = miopenRNNwithBias;
             break;
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenRNNBias"
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t miopenTohipRNNBias(miopenRNNBiasMode_t in,
                                hipdnnRNNBiasMode_t *out) {

    switch (in) {
        case miopenRNNNoBias:
            *out = HIPDNN_RNN_NO_BIAS;
            break;
        case miopenRNNwithBias:
            *out = HIPDNN_RNN_WITH_BIAS ;
             break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipRNNBias"
                              << in << ": NOT SUPPORTED." << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}
//=======================================================================================

// Accumulate Gradients Method to accumulate the dst and Prior with scaling
// factor beta

hipdnnStatus_t accumulateGradients(void *gradient, void *gradientPrior,
                                   hipdnnTensorDescriptor_t gradientDesc,
                                   const void *beta, hipdnnDataType_t *dataType) {
    // Trying to get the individual planes info

    int gradientArray[5];
    int gradientStride[5];

    CHECK_MIO(miopenGetTensorDescriptor((miopenTensorDescriptor_t)gradientDesc,
                                        (miopenDataType_t*)dataType, gradientArray,
                                        gradientStride));


    int totalElements = gradientArray[0] * gradientArray[1] * gradientArray[2] *
                        gradientArray[3];
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

   if(*dataType == miopenFloat) {

    float betaVal = *(static_cast<const float *>(beta));
    float *gradientF = static_cast<float *>(gradient);
    float *gradientPriorF = static_cast<float *>(gradientPrior);
    hipLaunchKernelGGL((TensorAdd<float>), dim3(blocks), dim3(threadsPerBlock),
                       0, 0, gradientF, gradientPriorF, betaVal, totalElements);
    CHECK_HIP(hipDeviceSynchronize());
    }
    else if (*dataType == miopenHalf){

    hc::half betaVal = *(static_cast<const hc::half *>(beta));
    hc::half *gradientF = static_cast<hc::half *>(gradient);
    hc::half *gradientPriorF = static_cast<hc::half *>(gradientPrior);
    hipLaunchKernelGGL((TensorAdd<hc::half>), dim3(blocks), dim3(threadsPerBlock),
                       0, 0, gradientF, gradientPriorF, betaVal, totalElements);
    CHECK_HIP(hipDeviceSynchronize());
    }

    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnCreate(hipdnnHandle_t *handle) {
    CHECK_MIO(miopenCreate((miopenHandle_t *)handle));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnDestroy(hipdnnHandle_t handle) {
    CHECK_MIO(miopenDestroy((miopenHandle_t)handle));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetStream(hipdnnHandle_t handle, hipdnnStream_t streamId) {
    CHECK_MIO(miopenSetStream((miopenHandle_t)handle,
                              (miopenAcceleratorQueue_t)streamId));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetStream(hipdnnHandle_t handle,
                               hipdnnStream_t *streamId) {
    CHECK_MIO(miopenGetStream((miopenHandle_t)handle,
                              (miopenAcceleratorQueue_t *)streamId));
    return HIPDNN_STATUS_SUCCESS;
}

size_t hipdnnGetVersion() { return 6000; }

//======================== Tensor and Operations ==============================

hipdnnStatus_t
hipdnnCreateTensorDescriptor( hipdnnTensorDescriptor_t *tensorDesc) {

    CHECK_MIO(miopenCreateTensorDescriptor((miopenTensorDescriptor_t*)tensorDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnTensorFormat_t format,
                                           hipdnnDataType_t dataType, int n,
                                           int c, int h, int w) {

    miopenDataType_t miDT;
    CHECK_HIPDNN(hipTensorFormatSupported(format));
    CHECK_HIPDNN(hipTomiopenDataType(dataType, &miDT));
    CHECK_MIO(miopenSet4dTensorDescriptor((miopenTensorDescriptor_t)tensorDesc,
                                          miDT, n, c, h, w));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetTensor4dDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnDataType_t *dataType, int *n,
                                           int *c, int *h, int *w, int *nStride,
                                           int *cStride, int *hStride,
                                           int *wStride) {
    miopenDataType_t midT;
    CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)tensorDesc,
                                          &midT, n, c, h, w,
                                          nStride, cStride, hStride, wStride));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnDestroyTensorDescriptor( hipdnnTensorDescriptor_t tensorDesc) {

    CHECK_MIO(miopenDestroyTensorDescriptor((miopenTensorDescriptor_t)tensorDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetTensor(hipdnnHandle_t handle,
                               const hipdnnTensorDescriptor_t yDesc, void *y,
                               const void *valuePtr) {

    CHECK_MIO(miopenSetTensor((miopenHandle_t)handle,
                              (miopenTensorDescriptor_t)yDesc, y, valuePtr));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------
// dstValue = alpha[0]*srcValue + beta[0]*priorDstValue

hipdnnStatus_t hipdnnAddTensor(hipdnnHandle_t handle, const void *alpha,
                               const hipdnnTensorDescriptor_t aDesc,
                               const void *A, const void *beta,
                               const hipdnnTensorDescriptor_t cDesc, void *C) {

    miopenTensorOp_t tensorOp = miopenTensorOpAdd;
    int alpha2 = 0;
    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, tensorOp, alpha,
                             (miopenTensorDescriptor_t)cDesc, C, alpha,
                             (miopenTensorDescriptor_t)aDesc, A, beta,
                             (miopenTensorDescriptor_t)cDesc, C));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnScaleTensor(hipdnnHandle_t handle,
                                 const hipdnnTensorDescriptor_t yDesc, void *y,
                                 const void *alpha) {

    CHECK_MIO(miopenScaleTensor((miopenHandle_t)handle,
                                (miopenTensorDescriptor_t)yDesc, y, alpha));
    return HIPDNN_STATUS_SUCCESS;
}

//============================ Tensor Operations ===============================

miopenTensorOp_t hipToMIOpenTensorOp(hipdnnOpTensorDescriptor_t opTensorDesc) {
    //TODO: Not needed, can be removed
    // int *result = reinterpret_cast<int *>(opTensorDesc);
    uintptr_t result = (uintptr_t)opTensorDesc;
    switch (result) {
        case 1:
            return miopenTensorOpAdd;
        case 2:
            return miopenTensorOpMul;
        case 3:
            return miopenTensorOpMin;
        case 4:
            return miopenTensorOpMax;
        default:
            return miopenTensorOpAdd;
    }
}

//------------------------------------------------------------------------------

hipdnnStatus_t miopenTohipOpTensorOp(miopenTensorOp_t in,
                                     hipdnnOpTensorOp_t *out) {
    switch (in) {
        case miopenTensorOpAdd:
            *out = HIPDNN_OP_TENSOR_ADD;
            break;
        case miopenTensorOpMul:
            *out = HIPDNN_OP_TENSOR_MUL;
            break;
        case miopenTensorOpMin:
            *out = HIPDNN_OP_TENSOR_MIN;
            break;
        case miopenTensorOpMax:
            *out = HIPDNN_OP_TENSOR_MAX;
            break;
        default:
            HIPDNN_OPEN_LOG_M("miopenTohipTensorOp " << in << ": NOT SUPPORTED."
                                                     << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipTomiopenOpTensorOp(hipdnnOpTensorOp_t in,
                                     miopenTensorOp_t *out) {
    switch (in) {
        case HIPDNN_OP_TENSOR_ADD:
            *out = miopenTensorOpAdd;
            break;
        case HIPDNN_OP_TENSOR_MUL:
            *out = miopenTensorOpMul;
            break;
        case HIPDNN_OP_TENSOR_MIN:
            *out = miopenTensorOpMin;
            break;
        case HIPDNN_OP_TENSOR_MAX:
            *out = miopenTensorOpMax;
            break;
        case HIPDNN_OP_TENSOR_SQRT:
        default:
            HIPDNN_OPEN_LOG_M("hipTomiopenTensorOp " << in << ": NOT SUPPORTED."
                                                     << std::flush);
            return HIPDNN_STATUS_NOT_SUPPORTED;
    }

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

typedef struct {
    hipdnnOpTensorOp_t opTensorOp;
    hipdnnDataType_t opTensorCompType;
    hipdnnNanPropagation_t opTensorNanOpt;
}structOpTensorDesc_t;

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnCreateOpTensorDescriptor(hipdnnOpTensorDescriptor_t *opTensorDesc) {

    *opTensorDesc = (void*)malloc(sizeof(structOpTensorDesc_t));
    CHECK_MALLOC(*opTensorDesc);

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnSetOpTensorDescriptor(hipdnnOpTensorDescriptor_t opTensorDesc,
                            hipdnnOpTensorOp_t opTensorOp,
                            hipdnnDataType_t opTensorCompType,
                            hipdnnNanPropagation_t opTensorNanOpt) {

    ((structOpTensorDesc_t*)opTensorDesc)->opTensorOp = opTensorOp;
    ((structOpTensorDesc_t*)opTensorDesc)->opTensorCompType = opTensorCompType;
    ((structOpTensorDesc_t*)opTensorDesc)->opTensorNanOpt = opTensorNanOpt;

    return HIPDNN_STATUS_SUCCESS;

}

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnGetOpTensorDescriptor(const hipdnnOpTensorDescriptor_t opTensorDesc,
                            hipdnnOpTensorOp_t *opTensorOp,
                            hipdnnDataType_t *opTensorCompType,
                            hipdnnNanPropagation_t *opTensorNanOpt) {

    *opTensorOp = ((structOpTensorDesc_t*)opTensorDesc)->opTensorOp;
    *opTensorCompType = ((structOpTensorDesc_t*)opTensorDesc)->opTensorCompType;
    *opTensorNanOpt = ((structOpTensorDesc_t*)opTensorDesc)->opTensorNanOpt;

    return HIPDNN_STATUS_SUCCESS;

}
//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnDestroyOpTensorDescriptor(hipdnnOpTensorDescriptor_t opTensorDesc) {

    free(opTensorDesc);
    return HIPDNN_STATUS_SUCCESS;

}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnOpTensor(
    hipdnnHandle_t handle, const hipdnnOpTensorDescriptor_t opTensorDesc,
    const void *alpha1, const hipdnnTensorDescriptor_t aDesc, const void *A,
    const void *alpha2, const hipdnnTensorDescriptor_t bDesc, const void *B,
    const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C) {

    miopenTensorOp_t miOpType;
    CHECK_HIPDNN( hipTomiopenOpTensorOp(
            ((structOpTensorDesc_t*)opTensorDesc)->opTensorOp, &miOpType ));

    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miOpType, alpha1,
                (miopenTensorDescriptor_t)aDesc, A, alpha2,
                (miopenTensorDescriptor_t)bDesc, B, beta,
                (miopenTensorDescriptor_t)cDesc, C));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnSetFilter4dDescriptor(hipdnnFilterDescriptor_t filterDesc,
                                           hipdnnTensorFormat_t format,
                                           hipdnnDataType_t dataType, int k,
                                           int c, int h, int w) {
    miopenDataType_t miDT;
    CHECK_HIPDNN(hipTensorFormatSupported(format));
    CHECK_HIPDNN(hipTomiopenDataType(dataType, &miDT));
    CHECK_MIO(miopenSet4dTensorDescriptor((miopenTensorDescriptor_t)filterDesc,
                                          miDT, k, c, h, w));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnCreateFilterDescriptor(
    hipdnnFilterDescriptor_t *filterDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateFilterDescriptor, " << filterDesc
                                                              << std::flush);
    // in miopen a filter descriptor is just a typedef to a tensor descriptor
    CHECK_HIPDNN(hipdnnCreateTensorDescriptor(filterDesc));
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateFilterDescriptor, " << filterDesc
                                                              << std::flush);
    return HIPDNN_STATUS_SUCCESS;
}

//============================= Convolution ====================================

/*
 * structConvDesc_t is used to contain cudnn-conv desc information that are not in miopenConvolutionDescriptor
 * hipdnnConvolutionDescriptor_t is just opaque pointer void*
 * Thus pointer structure `structConvDesc_t` is assigned in hipdnnConvolutionDescriptor_t
 * Structure also conatin hipdnnConvolutionDescriptor_t to hold descriptor
 * Use descriptor in structure with proper typecastings for MIopen API
 */

// structure to be used in place of convolution descriptor
typedef struct {
    hipdnnConvolutionDescriptor_t descriptor;
    hipdnnDataType_t convDataType;
    hipdnnMathType_t convMathType;
}structConvDesc_t;

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCreateConvolutionDescriptor(
    hipdnnConvolutionDescriptor_t *convDesc) {

    *convDesc = (void*)malloc(sizeof(structConvDesc_t));
    CHECK_MALLOC(*convDesc);
    hipdnnConvolutionDescriptor_t* convDesc_cast =
                            &( ((structConvDesc_t*)(*convDesc))->descriptor );
    CHECK_MIO(miopenCreateConvolutionDescriptor(
        (miopenConvolutionDescriptor_t *)convDesc_cast));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetConvolutionMathType(
    hipdnnConvolutionDescriptor_t convDesc, hipdnnMathType_t mathType) {

    HIPDNN_OPEN_LOG_I2( "Setting MathType by user is not supported in MIOpen."
                      << "Internally set based on datatype of input.");

    HIPDNN_OPEN_LOG_E("hipdnnSetConvolutionMathType"
                      << mathType << " NOT SUPPORTED in MIOpen"
                      << std::flush);

    ((structConvDesc_t*)(convDesc))->convMathType = mathType;

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetConvolution2dDescriptor(
    hipdnnConvolutionDescriptor_t convDesc, int pad_h, int pad_w, int u, int v,
    int upscalex, int upscaley, hipdnnConvolutionMode_t mode,
    hipdnnDataType_t computeType) {

    miopenConvolutionMode_t miConvMode;
    CHECK_HIPDNN(hipTomiopenConvolutionMode(mode,&miConvMode));
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO( miopenInitConvolutionDescriptor(
                                    (miopenConvolutionDescriptor_t)convDesc_cast,
                                    miConvMode, pad_h,
                                    pad_w, u, v, upscalex, upscaley));

    HIPDNN_OPEN_LOG_I2( "Setting MathType by user is not supported in MIOpen."
                      << "Internally set based on datatype of input.");

    ((structConvDesc_t*)(convDesc))->convDataType = computeType;

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetConvolutionNdDescriptor(
    hipdnnConvolutionDescriptor_t convDesc, int arrayLength, /* nbDims-2 size */
    const int padA[], const int filterStrideA[], const int dilationA[],
    hipdnnConvolutionMode_t mode,
    hipdnnDataType_t computeType)  // convolution data type
{
    HIPDNN_OPEN_LOG_C("Inside hipdnnSetConvolutionNdDescriptor with arrayLength:"
                        << arrayLength << std::flush);

    int pad_h, pad_w, u, v, d_h, d_w;

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;

    if (arrayLength == 2) {
        pad_h = padA[0];
        pad_w = padA[1];
        u = filterStrideA[0];
        v = filterStrideA[1];
        d_h = dilationA[0];
        d_w = dilationA[1];

        CHECK_MIO(miopenInitConvolutionDescriptor(
                                    (miopenConvolutionDescriptor_t)convDesc_cast,
                                    miopenConvolution, pad_h,
                                    pad_w, u, v, d_h, d_w) );
    }
    else if (arrayLength == 3) {
        // 3D convolution Scenario
        // Got to book keep additional padding, stride and dilation info along
        //  depth direction
        // Book keeping using global static std::map container
        // But first lets initialize the 2D Description
        pad_h = padA[0];
        pad_w = padA[1];
        u = filterStrideA[0];
        v = filterStrideA[1];
        d_h = dilationA[0];
        d_w = dilationA[1];
        CHECK_MIO(miopenInitConvolutionDescriptor(
                                    (miopenConvolutionDescriptor_t)convDesc_cast,
                                    miopenConvolution, pad_h,
                                    pad_w, u, v, d_h, d_w) );
        // Populate the map container with key being newly created 2Ddescriptor
        // and value a 3 dim array with index mapping as
        // 0-pad, 1-stride and 2-dilation
        int depthDesc[3];
        depthDesc[0] = padA[2];
        depthDesc[1] = filterStrideA[2];
        depthDesc[2] = dilationA[2];
        sDescTo3DConvolution[(miopenConvolutionDescriptor_t)convDesc_cast] =
            depthDesc;

    }
    else {
        HIPDNN_OPEN_LOG_E(
            "Inside hipdnnSetConvolutionNdDescriptor NOT SUPPORTED"
            << std::flush);
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolution2dDescriptor(
    const hipdnnConvolutionDescriptor_t convDesc, int *pad_h, int *pad_y,
    int *u, int *v, int *upscalex, int *upscaley, hipdnnConvolutionMode_t *mode,
    hipdnnDataType_t *computeType) {

    miopenConvolutionMode_t miMode;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenGetConvolutionDescriptor(
        (miopenConvolutionDescriptor_t)convDesc_cast, &miMode, pad_h, pad_y, u, v,
        upscalex, upscaley));

    CHECK_HIPDNN( miopenTohipdnnConvolutionMode(miMode, mode) );
    *computeType = ((structConvDesc_t*)(convDesc))->convDataType;

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolution2dForwardOutputDim(
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t inputTensorDesc,
    const hipdnnFilterDescriptor_t filterDesc, int *n, int *c, int *h, int *w) {

    HIPDNN_OPEN_LOG_C("HIPDNN_SOFTMAX_MODE_INSTANCE NOT SUPPORTED."
                      << std::flush);

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenGetConvolutionForwardOutputDim(
        (miopenConvolutionDescriptor_t)(convDesc_cast), // should be const in miopen.
        (miopenTensorDescriptor_t)inputTensorDesc,
        (miopenTensorDescriptor_t)filterDesc, n, c, h, w));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnDestroyConvolutionDescriptor(hipdnnConvolutionDescriptor_t convDesc) {
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenDestroyConvolutionDescriptor(
        (miopenConvolutionDescriptor_t)convDesc_cast));
    free(convDesc);

    return HIPDNN_STATUS_SUCCESS;
}

//-------------------------- Conv Forward --------------------------------------

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults) {

    size_t sizeInBytes = 0;
    void *sConvolutionForwardAlgorithmWorkspace;
    miopenConvFwdAlgorithm_t mialgo;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    // in miopen, workspace size does not depend on algo.
    CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)wDesc,
        (miopenTensorDescriptor_t)xDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)yDesc, &sizeInBytes));

    HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC hipdnnFindConvolutionForwardAlgorithm");

    CHECK_HIP(hipMalloc((void **)&sConvolutionForwardAlgorithmWorkspace,
                        sizeInBytes));

    size_t numBytes;
    void *x;
    void *y;
    void *w;

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)xDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&x, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)wDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&w, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)yDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&y, numBytes));

    CHECK_HIPDNN(hipdnnFindConvolutionForwardAlgorithmEx(
        handle, xDesc, x, wDesc, w, convDesc, yDesc,
        y, requestedAlgoCount, returnedAlgoCount, perfResults,
        sConvolutionForwardAlgorithmWorkspace, sizeInBytes));

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolutionForwardAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc,
    hipdnnConvolutionFwdPreference_t preference, size_t memoryLimitInBytes,
    hipdnnConvolutionFwdAlgo_t *algo) {
    miopenConvFwdAlgorithm_t mialgo;
    size_t sizeInBytes = 0;
    void *sConvolutionForwardAlgorithmWorkspace;
    // in miopen, workspace size does not depend on algo.

    if(preference == HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST)
        CHECK_HIPDNN(hipdnnGetConvolutionForwardWorkspaceSize(
            handle, xDesc, wDesc, convDesc, yDesc, *algo, &sizeInBytes));

    if(preference == HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT)
        sizeInBytes = memoryLimitInBytes;

    hipMalloc((void **)&sConvolutionForwardAlgorithmWorkspace, sizeInBytes);

    size_t numBytes;
    void *x;
    void *y;
    void *w;
    const int requestedAlgoCount = 1;
    int returnedAlgoCount;

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)xDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&x, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)wDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&w, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)yDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&y, numBytes));

    hipdnnConvolutionFwdAlgoPerf_t *perfResults =
        new hipdnnConvolutionFwdAlgoPerf_t[requestedAlgoCount];

    CHECK_HIPDNN(hipdnnFindConvolutionForwardAlgorithmEx(
        handle, xDesc, x, wDesc, w, convDesc, yDesc, y, requestedAlgoCount,
        &returnedAlgoCount, perfResults, sConvolutionForwardAlgorithmWorkspace,
        sizeInBytes));

    *algo = perfResults[0].algo;

    CHECK_HIP(hipFree(x));
    CHECK_HIP(hipFree(w));
    CHECK_HIP(hipFree(y));
    delete[] perfResults;
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFindConvolutionForwardAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, void *y, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionFwdAlgoPerf_t *perfResults,
    void *workSpace, size_t workSpaceSizeInBytes) {
    HIPDNN_OPEN_LOG_C("ENTER hipdnnFindConvolutionForwardAlgorithmEx: WS PTR"
                      << workSpace << ", " << workSpaceSizeInBytes
                      << std::flush);
    assert(x);
    assert(w);
    assert(y);
    miopenConvAlgoPerf_t *miopenPerfResults =
        new miopenConvAlgoPerf_t[requestedAlgoCount];

    HIPDNN_OPEN_LOG_C("Invoking miopenConvolutionForwardGetWorkSpaceSize"
                      << std::flush);
    size_t expectedWorkSpaceSize = 0, infoWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;

    workSpaceInternal = workSpace;
    expectedWorkSpaceSize = workSpaceSizeInBytes;

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenFindConvolutionForwardAlgorithm(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)xDesc, x,
        (miopenTensorDescriptor_t)wDesc, w,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)yDesc, y, requestedAlgoCount,
        returnedAlgoCount, miopenPerfResults, workSpaceInternal,
        expectedWorkSpaceSize, false  // exhaustiveSearch
        ));


    HIPDNN_OPEN_LOG_C("Invoked miopenFindConvolutionForwardAlgorithm");

    for (int i = 0; i < *returnedAlgoCount; i++) {
        CHECK_HIPDNN(miopenTohipConvolutionFwdAlgo(
            miopenPerfResults[i].fwd_algo, &(perfResults[i].algo)));
        perfResults[i].status =
            HIPDNN_STATUS_SUCCESS;  // TODO: miopen doesn't contain a 'status'
                                    // member variable , setting it to success
                                    // as of now.
        perfResults[i].time = miopenPerfResults[i].time;
        perfResults[i].memory = miopenPerfResults[i].memory;
    }

    delete[] miopenPerfResults;
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolutionForwardWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnFilterDescriptor_t wDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t yDesc, hipdnnConvolutionFwdAlgo_t algo,
    size_t *sizeInBytes) {
    *sizeInBytes = 0;

    HIPDNN_OPEN_LOG_C(
        "HIPDNN ENTER hipdnnGetConvolutionForwardWorkspaceSize, algo ="
        << algo << std::flush);

    miopenConvFwdAlgorithm_t mialgo;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    // in miopen, workspace size does not depend on algo.
    CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)wDesc,
        (miopenTensorDescriptor_t)xDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)yDesc, sizeInBytes));

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnConvolutionForward(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnConvolutionDescriptor_t convDesc,
    hipdnnConvolutionFwdAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const hipdnnTensorDescriptor_t yDesc, void *y) {
    HIPDNN_OPEN_LOG_C("calling hipdnnConvolutionForward." << std::flush);

    size_t expectedWorkSpaceSize = 0, infoWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;

        workSpaceInternal = workSpace;
        expectedWorkSpaceSize = workSpaceSizeInBytes;

    miopenConvFwdAlgorithm_t mialgo;
    CHECK_HIPDNN(hipTomiopenConvolutionFwdAlgo(algo, &mialgo));
    HIPDNN_OPEN_LOG_C("Invoked hipToMopenConvolutionFwdAlgo" << std::flush);
    HIPDNN_OPEN_LOG_C("Invoking MiopenConvolutionFwd" << std::flush);
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;

    if((*static_cast<const float *>(alpha)) == 1 && (*static_cast<const float *>(beta)) == 0) {

     CHECK_MIO(miopenConvolutionForward(
	    (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)xDesc, x,
        (miopenTensorDescriptor_t)wDesc, w,
        (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, beta,
        (miopenTensorDescriptor_t)yDesc, y, workSpaceInternal,
        expectedWorkSpaceSize));

    } else {
     void *dwPrior = SaveAsPriorBuffer(y);
     const float alpha1 = 1;
     const float beta1 = 0;

     CHECK_MIO(miopenConvolutionForward(
         (miopenHandle_t)handle, &alpha1, (miopenTensorDescriptor_t)xDesc, x,
         (miopenTensorDescriptor_t)wDesc, w,
         (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, &beta1,
         (miopenTensorDescriptor_t)yDesc, y, workSpaceInternal,
         expectedWorkSpaceSize));

     int alpha2 =0;
     CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                              (miopenTensorDescriptor_t)yDesc, y, beta,
                              (miopenTensorDescriptor_t)yDesc, dwPrior, &alpha2,
                              (miopenTensorDescriptor_t)yDesc, y));
     deallocPrior(dwPrior);
    }
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------ Conv Backward ---------------------------------------

hipdnnStatus_t hipdnnConvolutionBackwardBias(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const hipdnnTensorDescriptor_t dbDesc, void *db) {
    HIPDNN_OPEN_LOG_C("calling hipdnnConvolutionBackwardBias." << std::flush);

    CHECK_MIO(miopenConvolutionBackwardBias(
        (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)dyDesc, dy,
        beta, (miopenTensorDescriptor_t)dbDesc, db));

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults) {

    size_t sizeInBytes = 0;
    void *sConvolutionBackwardFilterAlgorithmWorkspace;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    // in miopen, workspace size does not depend on algo.
    CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc,
        (miopenTensorDescriptor_t)xDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)dwDesc, &sizeInBytes));

    HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC hipdnnFindConvolutionBackwardFilterAlgorithm");

    CHECK_HIP(hipMalloc((void **)&sConvolutionBackwardFilterAlgorithmWorkspace,
                        sizeInBytes));

    size_t numBytes;
    void *x;
    void *dy;
    void *dw;

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)xDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&x, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)dwDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&dw, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)dyDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&dy, numBytes));

    CHECK_HIPDNN(hipdnnFindConvolutionBackwardFilterAlgorithmEx(
        handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw, requestedAlgoCount,
        returnedAlgoCount, perfResults, sConvolutionBackwardFilterAlgorithmWorkspace,
        sizeInBytes));

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterAlgorithm(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc,
    hipdnnConvolutionBwdFilterPreference_t preference,
    size_t memoryLimitInBytes, hipdnnConvolutionBwdFilterAlgo_t *algo) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnGetConvolutionBackwardFilterAlgorithm ");

    size_t sizeInBytes = 0;
    void *sConvolutionBackwardFilterAlgorithmWorkspace;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    if(preference == HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST)
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc,
        (miopenTensorDescriptor_t)xDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)dwDesc, &sizeInBytes));

    if(preference == HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT)
        sizeInBytes = memoryLimitInBytes;

    HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC hipdnnGetConvolutionBackwardFilterAlgorithm");
    hipMalloc((void **)&sConvolutionBackwardFilterAlgorithmWorkspace, sizeInBytes);

    size_t numBytes;
    void *x;
    void *dy;
    void *dw;
    const int requestedAlgoCount = 1;
    int returnedAlgoCount;

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)xDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&x, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)dwDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&dw, numBytes));

    CHECK_MIO(
        miopenGetTensorNumBytes((miopenTensorDescriptor_t)dyDesc, &numBytes));
    CHECK_HIP(hipMalloc((void **)&dy, numBytes));

    hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults =
        new hipdnnConvolutionBwdFilterAlgoPerf_t[requestedAlgoCount];

    CHECK_HIPDNN(hipdnnFindConvolutionBackwardFilterAlgorithmEx(
        handle, xDesc, x, dyDesc, dy, convDesc, dwDesc, dw, requestedAlgoCount,
        &returnedAlgoCount, perfResults, sConvolutionBackwardFilterAlgorithmWorkspace,
        0));

    *algo = perfResults[0].algo;

    CHECK_HIP(hipFree(x));
    CHECK_HIP(hipFree(dw));
    CHECK_HIP(hipFree(dy));
    delete[] perfResults;
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFindConvolutionBackwardFilterAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc, void *dw,
    const int requestedAlgoCount, int *returnedAlgoCount,
    hipdnnConvolutionBwdFilterAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnFindConvolutionBackwardFilterAlgorithmEx");
    assert(x);
    assert(dy);
    assert(dw);

    miopenConvAlgoPerf_t *miopenPerfResults =
        new miopenConvAlgoPerf_t[requestedAlgoCount];

    size_t expectedWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;
    size_t infoWorkSpaceSize = 0;

    workSpaceInternal = workSpace;
    expectedWorkSpaceSize = workSpaceSizeInBytes;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    try {
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(
            (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)xDesc, x,
            (miopenConvolutionDescriptor_t)convDesc_cast,
            (miopenTensorDescriptor_t)dwDesc, dw, requestedAlgoCount,
            returnedAlgoCount, miopenPerfResults, workSpaceInternal,
            expectedWorkSpaceSize,
            false  // exhaustiveSearch
            ));

    } catch (std::exception &e) {
        std::cout << "EXCEPTION: hipdnnFindConvolutionBackwardFilterAlgorithmEx"
                  << e.what() << std::endl HIPDNNFLUSH
    }

    for (int i = 0; i < *returnedAlgoCount; i++) {
        CHECK_HIPDNN(miopenTohipConvolutionBwdFilterAlgo(
            miopenPerfResults[i].bwd_weights_algo, &(perfResults[i].algo)));
        perfResults[i].status =
            HIPDNN_STATUS_SUCCESS;  // TODO: miopen doesn't contain a 'status'
                                    // member variable , setting it to success
                                    // as of now.
        perfResults[i].time = miopenPerfResults[i].time;
        perfResults[i].memory = miopenPerfResults[i].memory;
    }
    delete[] miopenPerfResults;

    HIPDNN_OPEN_LOG_C("EXIT: hipdnnFindConvolutionBackwardFilterAlgorithmEx");

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolutionBackwardFilterWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnTensorDescriptor_t xDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnFilterDescriptor_t dwDesc,
    hipdnnConvolutionBwdFilterAlgo_t algo, size_t *sizeInBytes) {
    *sizeInBytes = 0;

    HIPDNN_OPEN_LOG_C(
        "ENTER hipdnnGetConvolutionBackwardFilterWorkspaceSize algo:"
        << algo << std::flush);
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc,
        (miopenTensorDescriptor_t)xDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)dwDesc, sizeInBytes));

    HIPDNN_OPEN_LOG_C("EXIT hipdnnGetConvolutionBackwardFilterWorkspaceSize:"
                      << *sizeInBytes << std::flush);

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnConvolutionBackwardFilter(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    hipdnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const hipdnnFilterDescriptor_t dwDesc, void *dw) {

    HIPDNN_OPEN_LOG_C("CALL_STACK: Inside hipdnnConvolutionBackwardFilter");
    size_t expectedWorkSpaceSize;
    void *workSpaceInternal = NULL;
    size_t infoWorkSpaceSize;


    workSpaceInternal = workSpace;
    expectedWorkSpaceSize = workSpaceSizeInBytes;

    int nbDimsRequested =1;
    int nbDims,dimA[1],strideA[1];
    hipdnnDataType_t dataType;
    hipdnnTensorFormat_t format = HIPDNN_TENSOR_NCHW;
    int filterDimA[1];

    hipdnnGetFilterNdDescriptor(dwDesc, nbDimsRequested, &dataType,
                                             &format, &nbDims, filterDimA);

    miopenConvBwdWeightsAlgorithm_t mialgo;
    CHECK_HIPDNN(hipTomiopenConvolutionBwdFilterAlgo(algo, &mialgo));
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    if (*static_cast<const float *>(beta) == 0) {
        CHECK_MIO(miopenConvolutionBackwardWeights(
            (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)xDesc, x,
            (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, beta,
            (miopenTensorDescriptor_t)dwDesc, dw, workSpaceInternal,
            expectedWorkSpaceSize));
    } else {
        const float tempBeta = 0;
        void *dwPrior = SaveAsPriorBuffer(dw);
        CHECK_MIO(miopenConvolutionBackwardWeights(
            (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)xDesc, x,
            (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, &tempBeta,
            (miopenTensorDescriptor_t)dwDesc, dw, workSpaceInternal,
            expectedWorkSpaceSize));
        accumulateGradients(dw, dwPrior, dwDesc, beta, &dataType);
        deallocPrior(dwPrior);
    }

    HIPDNN_OPEN_LOG_C("miopenConvolutionBackwardWeights "
                      << ",handle= " << handle << ",alpha=" << alpha
                      << ",xDesc=" << xDesc << ",x=" << x << ",dyDesc="
                      << dyDesc << ",dy=" << dy << ",convDesc=" << convDesc
                      << ",algo=" << algo << ",workSpace=" << workSpace
                      << ",workSpaceSizeInBytes = " << workSpaceSizeInBytes
                      << ",beta=" << beta << ",dwDesc=" << dwDesc
                      << ",dw=" << dw << std::flush);

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnGetConvolutionBackwardDataWorkspaceSize(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, hipdnnConvolutionBwdDataAlgo_t algo,
    size_t *sizeInBytes) {

    *sizeInBytes = 0;
    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    // does not depend on algo in miopen
    try {
        CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(
            (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc,
            (miopenTensorDescriptor_t)wDesc,
            (miopenConvolutionDescriptor_t)convDesc_cast,
            (miopenTensorDescriptor_t)dxDesc, sizeInBytes));
    } catch (std::exception &e) {
        std::cout
            << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
            << e.what() << std::endl HIPDNNFLUSH;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithm(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, const int requestedAlgoCount,
    int *returnedAlgoCount, hipdnnConvolutionBwdDataAlgoPerf_t *perfResults) {
    try {
        HIPDNN_OPEN_LOG_E(
            "ERROR: hipdnnFindConvolutionBackwardDataAlgorithm NOT IMPLEMENTED"
            << std::flush);

        return HIPDNN_STATUS_NOT_SUPPORTED;

    } catch (std::exception &e) {
        std::cout
            << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
            << e.what() << std::endl HIPDNNFLUSH;
    }
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetConvolutionBackwardDataAlgorithm(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc,
    const hipdnnTensorDescriptor_t dyDesc,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc,
    hipdnnConvolutionBwdDataPreference_t preference, size_t memoryLimitInBytes,
    hipdnnConvolutionBwdDataAlgo_t *algo) {
    try {
        HIPDNN_OPEN_LOG_C("Inside hipdnnGetConvolutionBackwardDataAlgorithm "
                          << std::flush);
        size_t numBytes;
        void *dx;
        void *dy;
        void *w;
        const int requestedAlgoCount = 1;
        int returnedAlgoCount;
        void *sConvolutionBackwardDataAlgorithmWorkspace = NULL;

        CHECK_MIO(miopenGetTensorNumBytes((miopenTensorDescriptor_t)dxDesc,
                                          &numBytes));
        CHECK_HIP(hipMalloc((void **)&dx, numBytes));

        CHECK_MIO(miopenGetTensorNumBytes((miopenTensorDescriptor_t)wDesc,
                                          &numBytes));
        CHECK_HIP(hipMalloc((void **)&w, numBytes));

        CHECK_MIO(miopenGetTensorNumBytes((miopenTensorDescriptor_t)dyDesc,
                                          &numBytes));
        CHECK_HIP(hipMalloc((void **)&dy, numBytes));

        hipdnnConvolutionBwdDataAlgoPerf_t *perfResults =
            new hipdnnConvolutionBwdDataAlgoPerf_t[requestedAlgoCount];

        CHECK_HIPDNN(hipdnnFindConvolutionBackwardDataAlgorithmEx(
            handle, wDesc, w, dyDesc, dy, convDesc, dxDesc, dx,
            requestedAlgoCount, &returnedAlgoCount, perfResults,
            sConvolutionBackwardDataAlgorithmWorkspace, 0));

        *algo = perfResults[0].algo;

        CHECK_HIP(hipFree(dx));
        CHECK_HIP(hipFree(w));
        CHECK_HIP(hipFree(dy));
        delete[] perfResults;

    } catch (std::exception &e) {
        std::cout
            << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
            << e.what() << std::endl HIPDNNFLUSH;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFindConvolutionBackwardDataAlgorithmEx(
    hipdnnHandle_t handle, const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t dxDesc, void *dx,
    const int requestedAlgoCount, int *returnedAlgoCount,
    hipdnnConvolutionBwdDataAlgoPerf_t *perfResults, void *workSpace,
    size_t workSpaceSizeInBytes) {
    HIPDNN_OPEN_LOG_C(
        "Inside hipdnnFindConvolutionBackwardDataAlgorithmEx: input ws size="
        << workSpaceSizeInBytes << ", requestedAlgoCount=" << requestedAlgoCount
        << ", WS PTR=" << workSpace << std::flush);

    size_t expectedWorkSpaceSize, infoWorkSpaceSize;
    void *workSpaceInternal = NULL;

    miopenConvAlgoPerf_t *miopenPerfResults =
        new miopenConvAlgoPerf_t[requestedAlgoCount];

        workSpaceInternal = workSpace;
        expectedWorkSpaceSize = workSpaceSizeInBytes;

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc,
        (miopenTensorDescriptor_t)wDesc,
        (miopenConvolutionDescriptor_t)convDesc_cast,
        (miopenTensorDescriptor_t)dxDesc, &infoWorkSpaceSize));

    try {
        CHECK_MIO(miopenFindConvolutionBackwardDataAlgorithm(
            (miopenHandle_t)handle, (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)wDesc, w,
            (miopenConvolutionDescriptor_t)convDesc_cast,
            (miopenTensorDescriptor_t)dxDesc, dx, requestedAlgoCount,
            returnedAlgoCount, miopenPerfResults, workSpaceInternal,
            expectedWorkSpaceSize,
            false  // exhaustiveSearch
            ));

        HIPDNN_OPEN_LOG_C(
            "...miopenFindConvolutionBackwardDataAlgorithm OK, "
            "returnedAlgoCount:"
            << *returnedAlgoCount << std::flush);
    } catch (std::exception &e) {
        std::cout
            << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
            << e.what() << std::endl HIPDNNFLUSH;
    }

    for (int i = 0; i < *returnedAlgoCount; i++) {
        CHECK_HIPDNN(miopenTohipConvolutionBwdDataAlgo(
            miopenPerfResults[i].bwd_data_algo, &(perfResults[i].algo)));

        perfResults[i].status =
            HIPDNN_STATUS_SUCCESS;  // TODO: miopen doesn't contain a 'status'
                                    // member variable , setting it to success
                                    // as of now.
        perfResults[i].time = miopenPerfResults[i].time;
        perfResults[i].memory = miopenPerfResults[i].memory;
    }

    delete[] miopenPerfResults;
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnConvolutionBackwardData(
    hipdnnHandle_t handle, const void *alpha,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnConvolutionDescriptor_t convDesc,
    hipdnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    HIPDNN_OPEN_LOG_C("ConvolutionBackwardData: WS PTR="
                      << workSpace << ", WS size = " << workSpaceSizeInBytes
                      << std::flush);

    size_t expectedWorkSpaceSize = 0;
    void *workSpaceInternal = NULL;
    size_t infoWorkSpaceSize = 0;

        workSpaceInternal = workSpace;
        expectedWorkSpaceSize = workSpaceSizeInBytes;

    int nbDimsRequested=1;
    int nbDims,dimA[1],strideA[1];
    hipdnnDataType_t dataType;
    hipdnnGetTensorNdDescriptor(dxDesc, nbDimsRequested, &dataType, &nbDims, dimA,strideA);

    try {
        // Allocate sConvolutionBackwardDataAlgorithmWorkspace to gather work
        // space value
        miopenConvBwdDataAlgorithm_t mialgo;
        CHECK_HIPDNN(hipTomiopenConvolutionBwdDataAlgo(algo, &mialgo));

        HIPDNN_OPEN_LOG_C(
            "ConvolutionBackwardData:  hipTomiopenConvolutionBwdDataAlgo OK."
            << std::flush);
        HIPDNN_OPEN_LOG_C(
            "ConvolutionBackwardData: about to invoke "
            "miopenConvolutionBackwardData."
            << ", WS PTR = " << workSpaceInternal
            << ", WS size =" << expectedWorkSpaceSize << std::flush);

        hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
        if (*static_cast<const float *>(beta) == 0) {
            CHECK_MIO(miopenConvolutionBackwardData(
                (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)dyDesc,
                dy, (miopenTensorDescriptor_t)wDesc, w,
                (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, beta,
                (miopenTensorDescriptor_t)dxDesc, dx, workSpaceInternal,
                expectedWorkSpaceSize));
        } else {
            HIPDNN_OPEN_LOG_C("Case Beta !=0." << std::flush);
            const float tempBeta = 0;
            void *dxPrior = SaveAsPriorBuffer(dx);
            CHECK_MIO(miopenConvolutionBackwardData(
                (miopenHandle_t)handle, alpha, (miopenTensorDescriptor_t)dyDesc,
                dy, (miopenTensorDescriptor_t)wDesc, w,
                (miopenConvolutionDescriptor_t)convDesc_cast, mialgo, &tempBeta,
                (miopenTensorDescriptor_t)dxDesc, dx, workSpaceInternal,
                expectedWorkSpaceSize));
            accumulateGradients(dx, dxPrior, dxDesc, beta, &dataType);
            deallocPrior(dxPrior);
        }

    } catch (std::exception &e) {
        std::cout
            << "Exception in hipdnnGetConvolutionBackwardDataWorkspaceSize: "
            << e.what() << std::endl HIPDNNFLUSH;
    }
    return HIPDNN_STATUS_SUCCESS;
}

//============================ Softmax layer ===================================

hipdnnStatus_t hipdnnSoftmaxForward(hipdnnHandle_t handle,
                                    hipdnnSoftmaxAlgorithm_t algo,
                                    hipdnnSoftmaxMode_t mode, const void *alpha,
                                    const hipdnnTensorDescriptor_t xDesc,
                                    const void *x, const void *beta,
                                    const hipdnnTensorDescriptor_t yDesc,
                                    void *y) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnSoftmaxForward");

    CHECK_HIPDNN(SoftmaxAlgorithmSupported(algo));
    CHECK_HIPDNN(hipSoftmaxModeSupported(mode));
    CHECK_MIO(miopenSoftmaxForward((miopenHandle_t)handle, alpha,
                                   (miopenTensorDescriptor_t)xDesc, x, beta,
                                   (miopenTensorDescriptor_t)yDesc, y));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSoftmaxBackward(
    hipdnnHandle_t handle, hipdnnSoftmaxAlgorithm_t algo,
    hipdnnSoftmaxMode_t mode, const void *alpha,
    const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnSoftmaxBackward");

    CHECK_HIPDNN(SoftmaxAlgorithmSupported(algo));
    CHECK_HIPDNN(hipSoftmaxModeSupported(mode));
    CHECK_MIO(miopenSoftmaxBackward((miopenHandle_t)handle, alpha,
                                    (miopenTensorDescriptor_t)yDesc, y,
                                    (miopenTensorDescriptor_t)dyDesc, dy, beta,
                                    (miopenTensorDescriptor_t)dxDesc, dx));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnCreatePoolingDescriptor(
    hipdnnPoolingDescriptor_t *poolingDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreatePoolingDescriptor");

    CHECK_MIO(miopenCreatePoolingDescriptor(
        (miopenPoolingDescriptor_t *)poolingDesc));
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnSetPooling2dDescriptor(
    hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t mode,
    hipdnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth,
    int verticalPadding, int horizontalPadding, int verticalStride,
    int horizontalStride) {
    miopenPoolingMode_t miPMode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnSetPooling2dDescriptor");

    CHECK_HIPDNN(hipTomiopenPoolingMode(mode, &miPMode));
    CHECK_MIO(miopenSet2dPoolingDescriptor(
        (miopenPoolingDescriptor_t)poolingDesc, miPMode, windowHeight,
        windowWidth, horizontalPadding, verticalPadding, horizontalStride,
        verticalStride));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dDescriptor(
    const hipdnnPoolingDescriptor_t poolingDesc, hipdnnPoolingMode_t *mode,
    hipdnnNanPropagation_t *maxpoolingNanOpt, int *windowHeight,
    int *windowWidth, int *verticalPadding, int *horizontalPadding,
    int *verticalStride, int *horizontalStride) {
    miopenPoolingMode_t mipmmode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnGetPooling2dDescriptor");

    CHECK_MIO(miopenGet2dPoolingDescriptor(
        (miopenPoolingDescriptor_t)poolingDesc, &mipmmode, windowHeight,
        windowWidth, horizontalPadding, horizontalPadding, horizontalStride,
        verticalStride));
    *maxpoolingNanOpt = HIPDNN_PROPAGATE_NAN;

    CHECK_HIPDNN(miopenTohipPoolingMode(mipmmode, mode));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnGetPooling2dForwardOutputDim(
    const hipdnnPoolingDescriptor_t poolingDesc,
    const hipdnnTensorDescriptor_t inputTensorDesc, int *n, int *c, int *h,
    int *w) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnGetPooling2dDescriptor");

    CHECK_MIO(miopenGetPoolingForwardOutputDim(
        (miopenPoolingDescriptor_t)poolingDesc,
        (miopenTensorDescriptor_t)inputTensorDesc, n, c, h, w));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyPoolingDescriptor(
    hipdnnPoolingDescriptor_t poolingDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnDestroyPoolingDescriptor");

    CHECK_MIO(
        miopenDestroyPoolingDescriptor((miopenPoolingDescriptor_t)poolingDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnPoolingForward(
    hipdnnHandle_t handle, const hipdnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y,
    bool do_backward) {

    int8_t *devptr = 0;

    size_t workSpaceSize = 0;

    HIPDNN_OPEN_LOG_C("Inside hipdnnPoolingForward");

    if (sDescToWorkspacePooling.find((miopenTensorDescriptor_t)yDesc) ==
        sDescToWorkspacePooling.end()) {
        // If the descriptor is not present in the bookkept map container,
        // create one and add to the container
        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: hipdnnPoolingForward");

        // the yDesc is used for the workspace, not the
        // poolingDesc
        CHECK_MIO(miopenPoolingGetWorkSpaceSize((miopenTensorDescriptor_t)yDesc,
                                                &workSpaceSize));
        CHECK_HIP(hipMalloc((void **)&devptr, workSpaceSize));
        sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc] = std::make_pair(devptr, workSpaceSize);

    } else {
        // Reuse the preallocated workspace
        devptr = sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc].first;
        workSpaceSize =
            sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc].second;
    }

    if((*static_cast<const float *>(alpha)) == 1 && (*static_cast<const float *>(beta)) == 0) {

        CHECK_MIO(miopenPoolingForward((miopenHandle_t)handle,
                                      (miopenPoolingDescriptor_t)poolingDesc,
                                      alpha, (miopenTensorDescriptor_t)xDesc, x,
                                      beta, (miopenTensorDescriptor_t)yDesc, y,
                                      do_backward,
                                      (void *)devptr, workSpaceSize));
	} else {

        void *dwPrior = SaveAsPriorBuffer(y);
        const float alpha1 = 1;
        const float beta1 = 0;
        CHECK_MIO(miopenPoolingForward((miopenHandle_t)handle,
                                       (miopenPoolingDescriptor_t)poolingDesc,
                                       &alpha1, (miopenTensorDescriptor_t)xDesc, x,
                                       &beta1, (miopenTensorDescriptor_t)yDesc, y,
                                       do_backward,
                                       (void *)devptr, workSpaceSize));
        int alpha2 =0;
        CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                                 (miopenTensorDescriptor_t)yDesc, y, beta,
                                 (miopenTensorDescriptor_t)yDesc, dwPrior, &alpha2,
                                 (miopenTensorDescriptor_t)yDesc, y));
        deallocPrior(dwPrior);
    }
    return HIPDNN_STATUS_SUCCESS;
}

//=================================!

hipdnnStatus_t hipdnnPoolingBackward(
    hipdnnHandle_t handle, const hipdnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    int8_t *devptr = 0;
    size_t workSpaceSize = 0;

    HIPDNN_OPEN_LOG_C("Inside hipdnnPoolingBackward");

    // forward and backward pooling can reuse tha same
    // map.

    if (sDescToWorkspacePooling.find((miopenTensorDescriptor_t)yDesc) ==
        sDescToWorkspacePooling.end()) {
        // yDesc is used for the workspace, not the
        // poolingDesc

        HIPDNN_OPEN_LOG_I("INTERNAL_ALLOC: hipdnnPoolingBackward");

        CHECK_MIO(miopenPoolingGetWorkSpaceSize((miopenTensorDescriptor_t)yDesc,
                                                &workSpaceSize));

        CHECK_HIP(hipMalloc((void **)&devptr, workSpaceSize));
        sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc] = std::make_pair(devptr, workSpaceSize);

    } else {
        devptr = sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc].first;
        workSpaceSize =
            sDescToWorkspacePooling[(miopenTensorDescriptor_t)yDesc].second;
    }

    if((*static_cast<const float *>(alpha)) == 1 && (*static_cast<const float *>(beta)) == 0) {

        CHECK_MIO(miopenPoolingBackward(
            (miopenHandle_t)handle, (miopenPoolingDescriptor_t)poolingDesc, alpha,
            (miopenTensorDescriptor_t)yDesc, y, (miopenTensorDescriptor_t)dyDesc,
            dy, (miopenTensorDescriptor_t)xDesc, x, beta,
            (miopenTensorDescriptor_t)dxDesc, dx,
            devptr));
    } else {

        void *dwPrior = SaveAsPriorBuffer(dx);
        const float alpha1 = 1;
        const float beta1 = 0;

        CHECK_MIO(miopenPoolingBackward(
            (miopenHandle_t)handle, (miopenPoolingDescriptor_t)poolingDesc, &alpha1,
            (miopenTensorDescriptor_t)yDesc, y, (miopenTensorDescriptor_t)dyDesc,
            dy, (miopenTensorDescriptor_t)xDesc, x, &beta1,
            (miopenTensorDescriptor_t)dxDesc, dx,
            devptr));

        int alpha2 =0;
        CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                                (miopenTensorDescriptor_t)dxDesc, dx, beta,
                                (miopenTensorDescriptor_t)dxDesc, dwPrior, &alpha2,
                                (miopenTensorDescriptor_t)dxDesc, dx));
        deallocPrior(dwPrior);
}
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnCreateActivationDescriptor(
    hipdnnActivationDescriptor_t *activationDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateActivationDescriptor");

    CHECK_MIO(miopenCreateActivationDescriptor(
        (miopenActivationDescriptor_t *)activationDesc));
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnSetActivationDescriptor(
    hipdnnActivationDescriptor_t activationDesc, // not const in cudnn
    hipdnnActivationMode_t mode,
    hipdnnNanPropagation_t reluNanOpt, double reluCeilingOrAlpha,
    double activBeta, double activExp) {
    miopenActivationMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnSetActivationDescriptor");

    CHECK_HIPDNN(hipTomiopenActivationMode(mode, &mimode));

    CHECK_MIO(miopenSetActivationDescriptor(
        static_cast<const miopenActivationDescriptor_t>(activationDesc), mimode,
        reluCeilingOrAlpha, activBeta, activExp));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnGetActivationDescriptor(
    const hipdnnActivationDescriptor_t activationDesc,
    hipdnnActivationMode_t *mode, hipdnnNanPropagation_t *reluNanOpt,
    double *reluCeilingOrAlpha, double *activBeta, double *activExp) {
    HIPDNN_OPEN_LOG_E("ENTER hipdnnGetActivationDescriptor");

    miopenActivationMode_t miactmode;

    CHECK_MIO(miopenGetActivationDescriptor(
        (miopenActivationDescriptor_t)activationDesc, &miactmode,
        reluCeilingOrAlpha, activBeta, activExp));

    CHECK_HIPDNN(miopenTohipActivationMode(miactmode, mode));
    *reluNanOpt = HIPDNN_PROPAGATE_NAN;
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyActivationDescriptor(
    hipdnnActivationDescriptor_t activationDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnDestroyActivationDescriptor");
    CHECK_MIO(miopenDestroyActivationDescriptor(
        (miopenActivationDescriptor_t)activationDesc));
    return HIPDNN_STATUS_SUCCESS;
}
//=================

hipdnnStatus_t hipdnnActivationForward(
    hipdnnHandle_t handle,
    hipdnnActivationDescriptor_t activationDesc,  // not const in cudnn
    const void *alpha, const hipdnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const hipdnnTensorDescriptor_t yDesc, void *y) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnActivationForward");

    void *dwPrior = SaveAsPriorBuffer(y);
    const float alpha1 = 1;
    const float beta1 = 0;
    CHECK_MIO(miopenActivationForward(
        (miopenHandle_t)handle, static_cast<const miopenActivationDescriptor_t>(activationDesc),
        &alpha1, (miopenTensorDescriptor_t)xDesc, x, &beta1,
        (miopenTensorDescriptor_t)yDesc, y));
    int alpha2 =0;
    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                             (miopenTensorDescriptor_t)yDesc, y, beta,
                             (miopenTensorDescriptor_t)yDesc, dwPrior, &alpha2,
                             (miopenTensorDescriptor_t)yDesc, y));
    deallocPrior(dwPrior);
    return HIPDNN_STATUS_SUCCESS;
}
//======================

hipdnnStatus_t hipdnnActivationBackward(
    hipdnnHandle_t handle,
    hipdnnActivationDescriptor_t activationDesc,  // const missing in cuda
    const void *alpha, const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnActivationBackward");

    void *dwPrior = SaveAsPriorBuffer(dx);
    const float alpha1 = 1;
    const float beta1 = 0;
    CHECK_MIO(miopenActivationBackward(
        (miopenHandle_t)handle, static_cast<const miopenActivationDescriptor_t>(activationDesc),
        &alpha1, (miopenTensorDescriptor_t)yDesc, y,
        (miopenTensorDescriptor_t)dyDesc, dy, (miopenTensorDescriptor_t)xDesc,
        x, &beta1, (miopenTensorDescriptor_t)dxDesc, dx));
    int alpha2 =0;
    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                             (miopenTensorDescriptor_t)dxDesc, dx, beta,
                             (miopenTensorDescriptor_t)dxDesc, dwPrior, &alpha2,
                             (miopenTensorDescriptor_t)dxDesc, dx));
    deallocPrior(dwPrior);

    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnCreateLRNDescriptor(hipdnnLRNDescriptor_t *normDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateLRNDescriptor");

    CHECK_MIO(miopenCreateLRNDescriptor((miopenLRNDescriptor_t *)normDesc));
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnSetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
                                      hipdnnLRNMode_t mode, unsigned lrnN,
                                      double lrnAlpha, double lrnBeta,
                                      double lrnK) {
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateLRNDescriptor");

    CHECK_HIPDNN(hipTomiopenLRNMode(mode, &mimode));
    CHECK_MIO(miopenSetLRNDescriptor((miopenLRNDescriptor_t)normDesc, mimode,
                                     lrnN, lrnAlpha, lrnBeta, lrnK));
    return HIPDNN_STATUS_SUCCESS;
}

//================

hipdnnStatus_t hipdnnGetLRNDescriptor(hipdnnLRNDescriptor_t normDesc,
                                      hipdnnLRNMode_t *mode, unsigned *lrnN,
                                      double *lrnAlpha, double *lrnBeta,
                                      double *lrnK) {
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateLRNDescriptor");

    CHECK_MIO(miopenGetLRNDescriptor((miopenLRNDescriptor_t)normDesc, &mimode,
                                     lrnN, lrnAlpha, lrnBeta, lrnK));

    CHECK_HIPDNN(miopenTohipLRNMode(mimode, mode));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnDestroyLRNDescriptor(hipdnnLRNDescriptor_t normDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnDestroyLRNDescriptor");

    CHECK_MIO(miopenDestroyLRNDescriptor((miopenLRNDescriptor_t)normDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelForward(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t yDesc, void *y, bool do_backward) {

    int8_t *devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnLRNCrossChannelForward");

    CHECK_HIPDNN(hipTomiopenLRNMode(lrnMode, &mimode));

	if( do_backward == 1) {
        if (sDescToWorkspaceLRN.find((miopenTensorDescriptor_t)yDesc) ==
            sDescToWorkspaceLRN.end()) {
            //yDesc is used for the workspace, not the
			//hipdnnLRNDescriptor_t

            CHECK_MIO(miopenLRNGetWorkSpaceSize((miopenTensorDescriptor_t)yDesc,
                                                &workSpaceSize));
            CHECK_HIP(hipMalloc((void **)&devptr, workSpaceSize));
            sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc] = std::make_pair(devptr, workSpaceSize);

        } else {
            devptr = sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc].first;
            workSpaceSize =
             sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc].second;
        }
	}

    void *dwPrior = SaveAsPriorBuffer(y);
    const float alpha1 = 1;
    const float beta1 = 0;

    CHECK_MIO(miopenLRNForward((miopenHandle_t)handle,
                               (miopenLRNDescriptor_t)normDesc, &alpha1,
                               (miopenTensorDescriptor_t)xDesc, x, &beta1,
                               (miopenTensorDescriptor_t)yDesc, y,
                               do_backward,
                               devptr));

    int alpha2 =0;
    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                             (miopenTensorDescriptor_t)yDesc, y, beta,
                             (miopenTensorDescriptor_t)yDesc, dwPrior, &alpha2,
                             (miopenTensorDescriptor_t)yDesc, y));
    deallocPrior(dwPrior);
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnLRNCrossChannelForwardEx(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t yDesc, void *y, size_t workspaceSize,
    void *workspace, bool do_backward) {
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnLRNCrossChannelForward");

    CHECK_HIPDNN(hipTomiopenLRNMode(lrnMode, &mimode));
    // mimode is otherwise unused.

    CHECK_MIO(miopenLRNForward((miopenHandle_t)handle,
                               (miopenLRNDescriptor_t)normDesc, alpha,
                               (miopenTensorDescriptor_t)xDesc, x, beta,
                               (miopenTensorDescriptor_t)yDesc, y,
                               do_backward,
                               workspace));

    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnLRNCrossChannelBackward(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx) {

    int8_t *devptr = 0;

    size_t workSpaceSize = 0;
    miopenStatus_t miStat;
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnLRNCrossChannelBackward");

    CHECK_HIPDNN(hipTomiopenLRNMode(lrnMode, &mimode));
    if (sDescToWorkspaceLRN.find((miopenTensorDescriptor_t)yDesc) ==
        sDescToWorkspaceLRN.end()) {
        // yDesc is used for the workspace, not the
        // hipdnnLRNDescriptor_t

        CHECK_MIO(miopenLRNGetWorkSpaceSize((miopenTensorDescriptor_t)yDesc,
                                            &workSpaceSize));
        CHECK_HIP(hipMalloc((void **)&devptr, workSpaceSize));
        sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc] = std::make_pair(devptr, workSpaceSize);

    } else {
        devptr = sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc].first;
        workSpaceSize =
            sDescToWorkspaceLRN[(miopenTensorDescriptor_t)yDesc].second;
    }

    CHECK_HIPDNN(hipdnnLRNCrossChannelBackwardEx(
        handle, normDesc, lrnMode, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta,
        dxDesc, dx, workSpaceSize, devptr));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnLRNCrossChannelBackwardEx(
    hipdnnHandle_t handle, hipdnnLRNDescriptor_t normDesc,
    hipdnnLRNMode_t lrnMode, const void *alpha,
    const hipdnnTensorDescriptor_t yDesc, const void *y,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const hipdnnTensorDescriptor_t dxDesc, void *dx,
    size_t workspacesize,  // HGSOS //NOTYET unused!!!
    void *workspace) {
    miopenLRNMode_t mimode;

    HIPDNN_OPEN_LOG_C("Inside hipdnnLRNCrossChannelBackwardEx");

    CHECK_HIPDNN(hipTomiopenLRNMode(lrnMode, &mimode));
    // mimode is otherwise unused.

    void *dwPrior = SaveAsPriorBuffer(dx);
    const float alpha1 = 1;
    const float beta1 = 0;

    CHECK_MIO(miopenLRNBackward(
        (miopenHandle_t)handle, (miopenLRNDescriptor_t)normDesc, &alpha1,
        (miopenTensorDescriptor_t)yDesc, y, (miopenTensorDescriptor_t)dyDesc,
        dy, (miopenTensorDescriptor_t)xDesc, x, &beta1,
        (miopenTensorDescriptor_t)dxDesc, dx, workspace));

    int alpha2 =0;
    CHECK_MIO(miopenOpTensor((miopenHandle_t)handle, miopenTensorOpAdd, alpha,
                             (miopenTensorDescriptor_t)dxDesc, dx, beta,
                             (miopenTensorDescriptor_t)dxDesc, dwPrior, &alpha2,
                             (miopenTensorDescriptor_t)dxDesc, dx));
    deallocPrior(dwPrior);
    return HIPDNN_STATUS_SUCCESS;
}

//==================================!

hipdnnStatus_t hipdnnDeriveBNTensorDescriptor(
    hipdnnTensorDescriptor_t derivedBnDesc,
    const hipdnnTensorDescriptor_t xDesc, hipdnnBatchNormMode_t mode) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnDeriveBNTensorDescriptor");

    miopenBatchNormMode_t miBNMode;
    CHECK_HIPDNN(hipTomiopenBatchNormMode(mode, &miBNMode));
    CHECK_MIO(miopenDeriveBNTensorDescriptor(
        (miopenTensorDescriptor_t)derivedBnDesc,
        (miopenTensorDescriptor_t)xDesc, miBNMode));
    return HIPDNN_STATUS_SUCCESS;
}

//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationForwardTraining(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode, void *alpha, void *beta,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t yDesc, void *y,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, void *bnScale,
    void *bnBias, double exponentialAverageFactor, void *resultRunningMean,
    void *resultRunningVariance, double epsilon, void *resultSaveMean,
    void *resultSaveInvVariance) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnBatchNormalizationForwardTraining");
    miopenBatchNormMode_t miBNMode;
    CHECK_HIPDNN(hipTomiopenBatchNormMode(mode, &miBNMode));
    CHECK_MIO(miopenBatchNormalizationForwardTraining(
        (miopenHandle_t)handle, miBNMode, alpha, beta,
        (miopenTensorDescriptor_t)xDesc, x, (miopenTensorDescriptor_t)yDesc, y,
        (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc, bnScale, bnBias,
        exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance));
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnnBatchNormalizationForwardInference(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode, void *alpha, void *beta,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t yDesc, void *y,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnBatchNormalizationForwardInference");
    miopenBatchNormMode_t miBNMode;
    CHECK_HIPDNN(hipTomiopenBatchNormMode(mode, &miBNMode));
    CHECK_MIO(miopenBatchNormalizationForwardInference(
        (miopenHandle_t)handle, miBNMode, alpha, beta,
        (miopenTensorDescriptor_t)xDesc, x, (miopenTensorDescriptor_t)yDesc, y,
        (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc,
        const_cast<void *>(bnScale), const_cast<void *>(bnBias),
        const_cast<void *>(estimatedMean),
        const_cast<void *>(estimatedVariance), epsilon));
    return HIPDNN_STATUS_SUCCESS;
}
//=============================================================================

hipdnnStatus_t hipdnnBatchNormalizationBackward(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode,
    const void *alphaDataDiff, const void *betaDataDiff,
    const void *alphaParamDiff, const void *betaParamDiff,
    const hipdnnTensorDescriptor_t xDesc, const void *x,
    const hipdnnTensorDescriptor_t dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t dxDesc, void *dx,
    const hipdnnTensorDescriptor_t bnScaleBiasDiffDesc, const void *bnScale,
    void *resultBnScaleDiff, void *resultBnBiasDiff, double epsilon,
    const void *savedMean, const void *savedInvVariance) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnBatchNormalizationBackward");

    miopenBatchNormMode_t miBNMode;
	int nbDimsRequested=1;
    int nbDims,dimA[1],strideA[1];
    hipdnnDataType_t dataType;
    hipdnnGetTensorNdDescriptor(xDesc, nbDimsRequested, &dataType, &nbDims, dimA,strideA);
    CHECK_HIPDNN(hipTomiopenBatchNormMode(mode, &miBNMode));
    if ((*static_cast<const float *>(betaDataDiff) == 0) &&
        (*static_cast<const float *>(betaParamDiff) == 0)) {
        HIPDNN_OPEN_LOG_I2("Accumulate Gradients is false");
        CHECK_MIO(miopenBatchNormalizationBackward(
            (miopenHandle_t)handle, miBNMode, alphaDataDiff, betaDataDiff,
            alphaParamDiff, betaParamDiff, (miopenTensorDescriptor_t)xDesc, x,
            (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)dxDesc, dx,
            (miopenTensorDescriptor_t)bnScaleBiasDiffDesc, bnScale,
            resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean,
            savedInvVariance));
        return HIPDNN_STATUS_SUCCESS;
    } else {
        HIPDNN_OPEN_LOG_I2("Accumulate Gradients is true");
        HIPDNN_OPEN_LOG_C(
            "Case where either betaDataDiff or betaParamDiff is nonzero");
        // Accumulate for resultBnScaleDiff
        const float tempBetaDataDiff = 0;
        const float tempBetaParamDiff = 0;
        void *dxPrior = SaveAsPriorBuffer(dx);
        void *resultBnScaleDiffPrior = SaveAsPriorBuffer(
            resultBnScaleDiff);  // Pointer to keep track of priorDst value
        void *resultBnBiasDiffPrior = SaveAsPriorBuffer(resultBnBiasDiff);
        CHECK_MIO(miopenBatchNormalizationBackward(
            (miopenHandle_t)handle, miBNMode, alphaDataDiff, &tempBetaDataDiff,
            alphaParamDiff, &tempBetaParamDiff, (miopenTensorDescriptor_t)xDesc,
            x, (miopenTensorDescriptor_t)dyDesc, dy,
            (miopenTensorDescriptor_t)dxDesc, dx,
            (miopenTensorDescriptor_t)bnScaleBiasDiffDesc, bnScale,
            resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean,
            savedInvVariance));
        accumulateGradients(dx, dxPrior, dxDesc, betaDataDiff, &dataType);
        accumulateGradients(resultBnScaleDiff, resultBnScaleDiffPrior,
                            bnScaleBiasDiffDesc, betaParamDiff, &dataType);
        accumulateGradients(resultBnBiasDiff, resultBnBiasDiffPrior,
                            bnScaleBiasDiffDesc, betaParamDiff, &dataType);
        deallocPrior(dxPrior);
        deallocPrior(resultBnBiasDiffPrior);
        deallocPrior(resultBnScaleDiffPrior);
    }

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetTensorNdDescriptor(hipdnnTensorDescriptor_t tensorDesc,
                                           hipdnnDataType_t dataType,
                                           int nbDims, const int dimA[],
                                           const int strideA[]) {
    miopenDataType_t moDT;
    HIPDNN_OPEN_LOG_C("ENTER: hipdnnSetTensorNdDescriptor "
                      << tensorDesc << "... nbDims=" << nbDims << std::flush);
    if (dataType != HIPDNN_DATA_FLOAT && dataType!= HIPDNN_DATA_HALF) {
        HIPDNN_OPEN_LOG_E(
            "ERROR: hipdnnSetTensorNdDescriptor only supports floats and half:"
            << dataType << std::flush);
        return HIPDNN_STATUS_NOT_SUPPORTED;

    } else {
        CHECK_HIPDNN(hipTomiopenDataType(dataType, &moDT));
        CHECK_MIO(miopenSetTensorDescriptor(
            (miopenTensorDescriptor_t)tensorDesc, moDT, nbDims,
            const_cast<int *>(dimA), const_cast<int *>(strideA)));
    }

    HIPDNN_OPEN_LOG_C("EXIT: hipdnnSetTensorNdDescriptor." << std::flush);
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetTensorNdDescriptor(
    const hipdnnTensorDescriptor_t tensorDesc, int nbDimsRequested,
    hipdnnDataType_t *dataType, int *nbDims, int dimA[], int strideA[]) {
    miopenDataType_t moDT;
    HIPDNN_OPEN_LOG_C("ENTER hipdnnGetTensorNdDescriptor " << tensorDesc
                                                           << std::flush);
    CHECK_MIO(miopenGetTensorDescriptor((miopenTensorDescriptor_t)tensorDesc,
                                        &moDT, dimA, strideA));

    CHECK_HIPDNN(miopenTohipDataType(moDT, dataType));
    CHECK_MIO(miopenGetTensorDescriptorSize(
        (miopenTensorDescriptor_t)tensorDesc, nbDims));
    HIPDNN_OPEN_LOG_C(
        "EXIT hipdnnGetTensorNdDescriptor, datatype  (miopen, hipdnn)= "
        << moDT << ", " << *dataType << ",size=" << *nbDims << std::flush);

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetFilterNdDescriptor(
    hipdnnFilterDescriptor_t filterDesc,
    hipdnnDataType_t dataType,  // image data type
    hipdnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
    miopenDataType_t moDT;
    HIPDNN_OPEN_LOG_C("ENTER hipdnnSetFilterNdDescriptor " << filterDesc
                                                           << std::flush);

    int strideA[nbDims - 1];

    for (int k = nbDims - 1; k >= 0; k--) {
        strideA[k] = (k != nbDims - 1) ? strideA[k + 1] * filterDimA[k + 1] : 1;
    }
    CHECK_HIPDNN(hipTomiopenDataType(dataType, &moDT));
    int strideDimA[nbDims - 1];
    for (int k = nbDims - 1; k >= 0; k--) {
        strideDimA[k] =
            (k != nbDims - 1) ? strideDimA[k + 1] * filterDimA[k + 1] : 1;
    }
    CHECK_MIO(miopenSetTensorDescriptor(
    	(miopenTensorDescriptor_t)filterDesc, moDT, nbDims,
	const_cast<int *>(filterDimA), const_cast<int *>(strideDimA)));
    HIPDNN_OPEN_LOG_C("EXIT hipdnnSetFilterNdDescriptor." << std::flush);
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetFilterNdDescriptor(
    const hipdnnFilterDescriptor_t filterDesc, int nbDimsRequested,
    hipdnnDataType_t *dataType,  // image data type
    hipdnnTensorFormat_t *format, int *nbDims, int filterDimA[]) {
    miopenDataType_t moDT;
    int strideDimA[nbDimsRequested];

    HIPDNN_OPEN_LOG_C("ENTER hipdnnGetFilterNdDescriptor " << filterDesc
                                                           << std::flush);
    CHECK_MIO(miopenGetTensorDescriptor((miopenTensorDescriptor_t)filterDesc,
                                        &moDT, filterDimA, strideDimA ));

    CHECK_HIPDNN(miopenTohipDataType(moDT, dataType));

    CHECK_MIO(miopenGetTensorDescriptorSize(
        (miopenTensorDescriptor_t)filterDesc, nbDims));
    *format = HIPDNN_TENSOR_NCHW;  // miopen defines only this format

    HIPDNN_OPEN_LOG_C("EXIT hipdnnGetFilterNdDescriptor");

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnDestroyFilterDescriptor(
    hipdnnFilterDescriptor_t filterDesc) {
    HIPDNN_OPEN_LOG_C("ENTER hipdnnDestroyFilterDescriptor " << filterDesc
                                                             << std::flush);
    CHECK_MIO(
        miopenDestroyTensorDescriptor((miopenTensorDescriptor_t)filterDesc));
    HIPDNN_OPEN_LOG_C("EXIT hipdnnDestroyFilterDescriptor." << std::flush);
    return HIPDNN_STATUS_SUCCESS;
}

// RNN APIs

hipdnnStatus_t hipdnnCreateRNNDescriptor(hipdnnRNNDescriptor_t *rnnDesc) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnCreateRNNDescriptor");
    CHECK_MIO(miopenCreateRNNDescriptor((miopenRNNDescriptor_t *)rnnDesc));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnDestroyRNNDescriptor(hipdnnRNNDescriptor_t rnnDesc) {
    CHECK_MIO(miopenDestroyRNNDescriptor((miopenRNNDescriptor_t)rnnDesc));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnCreatePersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                             const int minibatch,
                                             const hipdnnDataType_t dataType,
                                             hipdnnPersistentRNNPlan_t *plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetPersistentRNNPlan(hipdnnRNNDescriptor_t rnnDesc,
                                          hipdnnPersistentRNNPlan_t plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyPersistentRNNPlan(hipdnnPersistentRNNPlan_t plan) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v6(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, const int hiddenSize,
    const int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc,  // Between layers, not between recurrent steps.
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetRNNDescriptor( hipdnnHandle_t handle,
    hipdnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc,  // Between layers, not between recurrent steps.
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnRNNAlgo_t algo, hipdnnDataType_t dataType,
    hipdnnRNNBiasMode_t biasMode) {

    HIPDNN_OPEN_LOG_C("Inside hipdnnSetRNNDescriptor");

    miopenRNNInputMode_t inMode;
    miopenRNNDirectionMode_t miopendirection;
    miopenRNNMode_t rnnMode;
    miopenRNNAlgo_t algorithm;
    miopenDataType_t moDT;
    miopenRNNBiasMode_t moBT;

    CHECK_HIPDNN(hipTomiopenRNNInputMode(inputMode, &inMode));
    CHECK_HIPDNN(hipTomiopenRNNDirectionMode(direction, &miopendirection));
    CHECK_HIPDNN(hipTomiopenRNNMode(mode, &rnnMode));
    CHECK_HIPDNN(hipTomiopenRNNAlgo(algo, &algorithm));
    CHECK_HIPDNN(hipTomiopenDataType(dataType, &moDT));
    CHECK_HIPDNN(hipTomiopenRNNBias(biasMode, &moBT))

    CHECK_MIO(miopenSetRNNDescriptor((miopenRNNDescriptor_t) rnnDesc,
        hiddenSize, numLayers, inMode, miopendirection, rnnMode,  moBT,
        algorithm, moDT));
}

hipdnnStatus_t hipdnnSetRNNDescriptor_v5(
    hipdnnRNNDescriptor_t rnnDesc, int hiddenSize, int numLayers,
    hipdnnDropoutDescriptor_t
        dropoutDesc, /* Between layers, not between remorrent steps. */
    hipdnnRNNInputMode_t inputMode, hipdnnDirectionMode_t direction,
    hipdnnRNNMode_t mode, hipdnnDataType_t dataType) {
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetRNNDescriptor(hipdnnHandle_t handle,
    hipdnnRNNDescriptor_t rnnDesc, int* hiddenSize, int* numLayers,
    hipdnnDropoutDescriptor_t *dropoutDesc, hipdnnRNNInputMode_t *inputMode,
    hipdnnDirectionMode_t *direction, hipdnnRNNMode_t *mode, hipdnnRNNAlgo_t *algo,
    hipdnnDataType_t *dataType, hipdnnRNNBiasMode_t *biasMode) {

    miopenRNNInputMode_t moRIM;
    miopenRNNDirectionMode_t moDM;
    miopenRNNMode_t moRM;
    miopenDataType_t moDT;
    miopenRNNAlgo_t moRA;
    miopenRNNBiasMode_t moBM;

    CHECK_MIO(miopenGetRNNDescriptor((miopenRNNDescriptor_t) rnnDesc, &moRM, &moRA, &moRIM, &moDM,
        &moBM, hiddenSize, numLayers));

    CHECK_HIPDNN(miopenTohipRNNInputMode(moRIM, inputMode));
    CHECK_HIPDNN(miopenTohipRNNDirectionMode(moDM, direction));
    CHECK_HIPDNN(miopenTohipRNNAlgo(moRA, algo));
    CHECK_HIPDNN(miopenTohipDataType(moDT, dataType));
    CHECK_HIPDNN(miopenTohipRNNMode(moRM, mode));
    CHECK_HIPDNN(miopenTohipRNNBias(moBM, biasMode));
}

hipdnnStatus_t hipdnnGetRNNLayerParamSize(hipdnnHandle_t handle,
                                        hipdnnRNNDescriptor_t rnnDesc,
                                        const int layer,
                                        hipdnnTensorDescriptor_t xDesc,
                                        const int paramID,
                                        size_t *numBytes) {
    CHECK_MIO(miopenGetRNNLayerParamSize(
        (miopenHandle_t) handle, (miopenRNNDescriptor_t) rnnDesc,
        layer, (miopenTensorDescriptor_t) xDesc, paramID, numBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNLayerBiasSize(hipdnnHandle_t handle,
                                        hipdnnRNNDescriptor_t rnnDesc,
                                        const int layer,
                                        const int biasID,
                                        size_t *numBytes) {

    CHECK_MIO(miopenGetRNNLayerBiasSize((miopenHandle_t) handle, (miopenRNNDescriptor_t) rnnDesc, layer,
                    biasID, numBytes));
    return HIPDNN_STATUS_SUCCESS;
}


hipdnnStatus_t hipdnnGetRNNWorkspaceSize(hipdnnHandle_t handle,
                                         const hipdnnRNNDescriptor_t rnnDesc,
                                         const int seqLength,
                                         const hipdnnTensorDescriptor_t *xDesc,
                                         size_t *sizeInBytes) {
    CHECK_MIO(miopenGetRNNWorkspaceSize(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength,
        (miopenTensorDescriptor_t *)xDesc, sizeInBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNTrainingReserveSize(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc,
    size_t *sizeInBytes) {
    CHECK_MIO(miopenGetRNNTrainingReserveSize(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength,
        (miopenTensorDescriptor_t *)xDesc, sizeInBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNParamsSize(hipdnnHandle_t handle,
                                      const hipdnnRNNDescriptor_t rnnDesc,
                                      const hipdnnTensorDescriptor_t xDesc,
                                      size_t *sizeInBytes,
                                      hipdnnDataType_t dataType) {

    miopenDataType_t moDT;
    CHECK_HIPDNN(hipTomiopenDataType(dataType, &moDT));

    CHECK_MIO(miopenGetRNNParamsSize((miopenHandle_t)handle,
        static_cast<miopenRNNDescriptor_t> (rnnDesc),
        static_cast<miopenTensorDescriptor_t> (xDesc), sizeInBytes, moDT));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNLinLayerMatrixParams(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc, const int layer,
    const hipdnnTensorDescriptor_t xDesc, const hipdnnFilterDescriptor_t wDesc,
    const void *w, const int linLayerID,
    hipdnnFilterDescriptor_t linLayerMatDesc, void **linLayerMat) {

    CHECK_MIO(miopenGetRNNLayerParam(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, layer,
        (miopenTensorDescriptor_t) xDesc, (miopenTensorDescriptor_t) wDesc, w,
        linLayerID, (miopenTensorDescriptor_t) linLayerMatDesc, linLayerMat));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNLinLayerBiasParams(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc, const int layer,
    const hipdnnTensorDescriptor_t xDesc, const hipdnnFilterDescriptor_t wDesc,
    const void *w, const int linLayerID,
    hipdnnFilterDescriptor_t linLayerBiasDesc, void **linLayerBias) {

    CHECK_MIO(miopenGetRNNLayerBias((miopenHandle_t) handle,
        (miopenRNNDescriptor_t) rnnDesc, layer,
        static_cast<miopenTensorDescriptor_t> (xDesc),
        static_cast<miopenTensorDescriptor_t> (wDesc), w, linLayerID,
        (miopenTensorDescriptor_t) linLayerBiasDesc, *linLayerBias));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNParamsDescriptor(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, hipdnnTensorDescriptor_t xDesc,
    hipdnnTensorDescriptor_t wDesc, hipdnnDataType_t dtype) {

    miopenDataType_t moDT;
    CHECK_HIPDNN(hipTomiopenDataType(dtype, &moDT));
    CHECK_MIO(miopenGetRNNParamsDescriptor((miopenHandle_t) handle,
            (miopenRNNDescriptor_t) rnnDesc, (miopenTensorDescriptor_t) xDesc,
            (miopenTensorDescriptor_t) wDesc, moDT));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnGetRNNInputTensorSize(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, const int seqLen,
    hipdnnTensorDescriptor_t *xDesc, size_t *numBytes) {

    CHECK_MIO(miopenGetRNNInputTensorSize((miopenHandle_t) handle,
            (miopenRNNDescriptor_t) rnnDesc, seqLen, (miopenTensorDescriptor_t *) xDesc,
            numBytes));

    return HIPDNN_STATUS_SUCCESS;
}
hipdnnStatus_t hipdnnGetRNNHiddenTensorSize(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, const int seqLen,
    hipdnnTensorDescriptor_t *xDesc, size_t *numBytes) {

    CHECK_MIO(miopenGetRNNHiddenTensorSize((miopenHandle_t) handle,
           (miopenRNNDescriptor_t) rnnDesc, seqLen, (miopenTensorDescriptor_t* )xDesc,
            numBytes));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetRNNLayerParam(
    hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc, const int layer,
    hipdnnTensorDescriptor_t xDesc, hipdnnTensorDescriptor_t wDesc, void *w,
    const int paramID, hipdnnTensorDescriptor_t paramDesc, const void *layerParam) {

    CHECK_MIO(miopenSetRNNLayerParam((miopenHandle_t) handle,
            (miopenRNNDescriptor_t) rnnDesc, layer, (miopenTensorDescriptor_t) xDesc,
            (miopenTensorDescriptor_t) wDesc, w, paramID, (miopenTensorDescriptor_t) paramDesc,
            layerParam));

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetRNNLayerBias(hipdnnHandle_t handle, hipdnnRNNDescriptor_t rnnDesc,
    const int layer, hipdnnTensorDescriptor_t xDesc, hipdnnTensorDescriptor_t wDesc, void *w,
    const int biasID, hipdnnTensorDescriptor_t biasDesc, const void *layerBias) {

    CHECK_MIO(miopenSetRNNLayerBias((miopenHandle_t) handle, (miopenRNNDescriptor_t) rnnDesc,
            layer, (miopenTensorDescriptor_t) xDesc, (miopenTensorDescriptor_t) wDesc, w, biasID,
            (miopenTensorDescriptor_t) biasDesc, layerBias));;

    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnRNNForwardInference(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t cxDesc, const void *cx,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t *yDesc, void *y,
    const hipdnnTensorDescriptor_t hyDesc, void *hy,
    const hipdnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes) {

    CHECK_MIO(miopenRNNForwardInference(
        (miopenHandle_t) handle, static_cast<miopenRNNDescriptor_t> (rnnDesc),
        seqLength, (miopenTensorDescriptor_t *)xDesc, x,
        (miopenTensorDescriptor_t) hxDesc, hx, (miopenTensorDescriptor_t) cxDesc, cx,
        (miopenTensorDescriptor_t) wDesc, w, (miopenTensorDescriptor_t *) yDesc, y,
        (miopenTensorDescriptor_t) hyDesc, hy, (miopenTensorDescriptor_t) cyDesc, cy,
        workspace, workSpaceSizeInBytes));

    return HIPDNN_STATUS_SUCCESS;

}

hipdnnStatus_t hipdnnRNNForwardTraining(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t cxDesc, const void *cx,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t *yDesc, void *y,
    const hipdnnTensorDescriptor_t hyDesc, void *hy,
    const hipdnnTensorDescriptor_t cyDesc, void *cy, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    CHECK_MIO(miopenRNNForwardTraining(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength,
        (miopenTensorDescriptor_t *)xDesc, x, (miopenTensorDescriptor_t)hxDesc,
        hx, (miopenTensorDescriptor_t)cxDesc, cx,
        (miopenTensorDescriptor_t)wDesc, w, (miopenTensorDescriptor_t *)yDesc,
        y, (miopenTensorDescriptor_t)hyDesc, hy,
        (miopenTensorDescriptor_t)cyDesc, cy, workspace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnRNNBackwardData(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *yDesc, const void *y,
    const hipdnnTensorDescriptor_t *dyDesc, const void *dy,
    const hipdnnTensorDescriptor_t dhyDesc, const void *dhy,
    const hipdnnTensorDescriptor_t dcyDesc, const void *dcy,
    const hipdnnFilterDescriptor_t wDesc, const void *w,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t cxDesc, const void *cx,
    const hipdnnTensorDescriptor_t *dxDesc, void *dx,
    const hipdnnTensorDescriptor_t dhxDesc, void *dhx,
    const hipdnnTensorDescriptor_t dcxDesc, void *dcx, void *workspace,
    size_t workSpaceSizeInBytes, void *reserveSpace,
    size_t reserveSpaceSizeInBytes) {
    CHECK_MIO(miopenRNNBackwardData(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength,
        (miopenTensorDescriptor_t *)yDesc, y,
        (miopenTensorDescriptor_t *)dyDesc, dy,
        (miopenTensorDescriptor_t)dhyDesc, dhy,
        (miopenTensorDescriptor_t)dcyDesc, dcy, (miopenTensorDescriptor_t)wDesc,
        w, (miopenTensorDescriptor_t)hxDesc, hx,
        (miopenTensorDescriptor_t)cxDesc, cx,
        (miopenTensorDescriptor_t *)dxDesc, dx,
        (miopenTensorDescriptor_t)dhxDesc, dhx,
        (miopenTensorDescriptor_t)dcxDesc, dcx, workspace, workSpaceSizeInBytes,
        reserveSpace, reserveSpaceSizeInBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnRNNBackwardWeights(
    hipdnnHandle_t handle, const hipdnnRNNDescriptor_t rnnDesc,
    const int seqLength, const hipdnnTensorDescriptor_t *xDesc, const void *x,
    const hipdnnTensorDescriptor_t hxDesc, const void *hx,
    const hipdnnTensorDescriptor_t *yDesc, const void *y, const void *workspace,
    size_t workSpaceSizeInBytes, const hipdnnFilterDescriptor_t dwDesc,
    void *dw, const void *reserveSpace, size_t reserveSpaceSizeInBytes) {
    CHECK_MIO(miopenRNNBackwardWeights(
        (miopenHandle_t)handle, (miopenRNNDescriptor_t)rnnDesc, seqLength,
        (miopenTensorDescriptor_t *)xDesc, x, (miopenTensorDescriptor_t)hxDesc,
        hx, (miopenTensorDescriptor_t *)yDesc, y,
        (miopenTensorDescriptor_t)dwDesc, dw, const_cast<void *>(workspace),
        workSpaceSizeInBytes, reserveSpace, reserveSpaceSizeInBytes));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnSetPoolingNdDescriptor(
    hipdnnPoolingDescriptor_t poolingDesc, const hipdnnPoolingMode_t mode,
    const hipdnnNanPropagation_t maxpoolingNanOpt, int nbDims,
    const int windowDimA[], const int paddingA[], const int strideA[]) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnSetPoolingNdDescriptor with nbDims :"
                      << nbDims

                      << std::flush);
    if (nbDims == 2) {
        // 2D Pooling
        int windowHeight = windowDimA[0];
        int windowWidth = windowDimA[1];
        int pad_h = paddingA[0];
        int pad_w = paddingA[1];
        int u = strideA[0];
        int v = strideA[1];
        miopenPoolingMode_t pooling_mode;
        CHECK_HIPDNN(hipTomiopenPoolingMode(mode, &pooling_mode));
        CHECK_MIO(miopenSet2dPoolingDescriptor(
            (miopenPoolingDescriptor_t)poolingDesc, pooling_mode, windowHeight,
            windowWidth, pad_h, pad_w, u, v));
    } else {
        HIPDNN_OPEN_LOG_E("Higher dimensions > 2 Pooling is not supported"
                          << std::flush);
        return HIPDNN_STATUS_NOT_SUPPORTED;
    }
    return HIPDNN_STATUS_SUCCESS;
}

// human-readable error messages
// hipdnnGetErrorString
const char *hipdnnGetErrorString(hipdnnStatus_t status) {
    switch (status) {
        case HIPDNN_STATUS_SUCCESS:
            return "HIPDNN_STATUS_SUCCESS";

        case HIPDNN_STATUS_NOT_INITIALIZED:
            return "HIPDNN_STATUS_NOT_INITIALIZED";

        case HIPDNN_STATUS_ALLOC_FAILED:
            return "HIPDNN_STATUS_ALLOC_FAILED";

        case HIPDNN_STATUS_BAD_PARAM:
            return "HIPDNN_STATUS_BAD_PARAM";

        case HIPDNN_STATUS_INTERNAL_ERROR:
            return "HIPDNN_STATUS_INTERNAL_ERROR";

        case HIPDNN_STATUS_INVALID_VALUE:
            return "HIPDNN_STATUS_INVALID_VALUE";

        case HIPDNN_STATUS_ARCH_MISMATCH:
            return "HIPDNN_STATUS_ARCH_MISMATCH";

        case HIPDNN_STATUS_MAPPING_ERROR:
            return "HIPDNN_STATUS_MAPPING_ERROR";

        case HIPDNN_STATUS_EXECUTION_FAILED:
            return "HIPDNN_STATUS_EXECUTION_FAILED";

        case HIPDNN_STATUS_NOT_SUPPORTED:
            return "HIPDNN_STATUS_NOT_SUPPORTED";

        case HIPDNN_STATUS_LICENSE_ERROR:
            return "HIPDNN_STATUS_LICENSE_ERROR";

        case HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return "HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";

        default:
            return "Unrecognized Status Code";
    }
}

hipdnnStatus_t hipdnnBatchNormalizationForwardInference(
    hipdnnHandle_t handle, hipdnnBatchNormMode_t mode,
    const void *alpha,  // alpha[0] = result blend factor
    const void *beta,   // beta[0] = dest layer blend factor
    const hipdnnTensorDescriptor_t xDesc,
    const void *x,  // NxCxHxW
    const hipdnnTensorDescriptor_t yDesc,
    void *y,  // NxCxHxW
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    HIPDNN_OPEN_LOG_C("Inside hipdnnBatchNormalizationForwardInference");
    miopenBatchNormMode_t miBNMode;
    CHECK_HIPDNN(hipTomiopenBatchNormMode(mode, &miBNMode));
    CHECK_MIO(miopenBatchNormalizationForwardInference(
        (miopenHandle_t)handle, miBNMode, const_cast<void *>(alpha),
        const_cast<void *>(beta), (miopenTensorDescriptor_t)xDesc, x,
        (miopenTensorDescriptor_t)yDesc, y,
        (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc,
        const_cast<void *>(bnScale), const_cast<void *>(bnBias),
        const_cast<void *>(estimatedMean),
        const_cast<void *>(estimatedVariance), epsilon));
    return HIPDNN_STATUS_SUCCESS;
}

hipdnnStatus_t hipdnnCreateDropoutDescriptor(
    hipdnnDropoutDescriptor_t *dropoutDesc) {
    HIPDNN_OPEN_LOG_E("hipdnnCreateDropoutDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetDropoutDescriptor(hipdnnDropoutDescriptor_t dropoutDesc,
                                          hipdnnHandle_t handle, float dropout,
                                          void *states, size_t stateSizeInBytes,
                                          unsigned long long seed) {
    HIPDNN_OPEN_LOG_E("hipdnnSetDropoutDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDropoutGetStatesSize(hipdnnHandle_t handle,
                                          size_t *sizeInBytes) {
    HIPDNN_OPEN_LOG_E("hipdnnDropoutGetStatesSize: NOT SUPPORTED."
                      << std::endl
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyDropoutDescriptor(
    hipdnnDropoutDescriptor_t dropoutDesc) {
    HIPDNN_OPEN_LOG_E("hipdnnDestroyDropoutDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnCreateReduceTensorDescriptor(
    hipdnnReduceTensorDescriptor_t *reduceTensorDesc) {
    HIPDNN_OPEN_LOG_E("hipdnnCreateReduceTensorDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetTensor4dDescriptorEx(
    hipdnnTensorDescriptor_t tensorDesc,
    hipdnnDataType_t dataType, /* image data type */
    int n,                     /* number of inputs (batch size) */
    int c,                     /* number of input feature maps */
    int h,                     /* height of input section */
    int w,                     /* width of input section */
    int nStride, int cStride, int hStride, int wStride) {
    HIPDNN_OPEN_LOG_E("hipdnnSetTensor4dDescriptorEx: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnSetReduceTensorDescriptor(
    hipdnnReduceTensorDescriptor_t reduceTensorDesc,
    hipdnnReduceTensorOp_t reduceTensorOp,
    hipdnnDataType_t reduceTensorCompType,
    hipdnnNanPropagation_t reduceTensorNanOpt,
    hipdnnReduceTensorIndices_t reduceTensorIndices,
    hipdnnIndicesType_t reduceTensorIndicesType) {
    HIPDNN_OPEN_LOG_E("hipdnnSetReduceTensorDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnGetReductionWorkspaceSize(
    hipdnnHandle_t handle,
    const hipdnnReduceTensorDescriptor_t reduceTensorDesc,
    const hipdnnTensorDescriptor_t aDesc, const hipdnnTensorDescriptor_t cDesc,
    size_t *sizeInBytes) {
    HIPDNN_OPEN_LOG_E("hipdnnGetReductionWorkspaceSize: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnReduceTensor(
    hipdnnHandle_t handle,
    const hipdnnReduceTensorDescriptor_t reduceTensorDesc, void *indices,
    size_t indicesSizeInBytes, void *workspace, size_t workspaceSizeInBytes,
    const void *alpha, const hipdnnTensorDescriptor_t aDesc, const void *A,
    const void *beta, const hipdnnTensorDescriptor_t cDesc, void *C) {
    HIPDNN_OPEN_LOG_E("hipdnnReduceTensor: NOT SUPPORTED." << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}

hipdnnStatus_t hipdnnDestroyReduceTensorDescriptor(
    hipdnnReduceTensorDescriptor_t reduceTensorDesc) {
    HIPDNN_OPEN_LOG_E("hipdnnDestroyReduceTensorDescriptor: NOT SUPPORTED."
                      << std::flush);
    return HIPDNN_STATUS_NOT_SUPPORTED;
}


 hipdnnStatus_t hipdnnSetConvolutionGroupCount(
    hipdnnConvolutionDescriptor_t convDesc, int groupCount ) {

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(miopenSetConvolutionGroupCount(
        (miopenConvolutionDescriptor_t)convDesc_cast, groupCount) );
    return HIPDNN_STATUS_SUCCESS;

}

//============================ MIO-Fusion ======================================

hipdnnStatus_t
hipdnnCreateFusionPlan(hipdnnFusionPlanDescriptor_t *fusePlanDesc,
                       const hipdnnFusionDirection_t fuseDirection,
                       const hipdnnTensorDescriptor_t inputDesc) {
    CHECK_MIO(
        miopenCreateFusionPlan((miopenFusionPlanDescriptor_t *)fusePlanDesc,
                               (miopenFusionDirection_t)fuseDirection,
                               (miopenTensorDescriptor_t)inputDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFusionPlanGetOp(hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                     const int op_idx,
                                     hipdnnFusionOpDescriptor_t *op) {
    CHECK_MIO(miopenFusionPlanGetOp((miopenFusionPlanDescriptor_t)fusePlanDesc,
                                    op_idx, (miopenFusionOpDescriptor_t *)op));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFusionPlanGetWorkSpaceSize(
    hipdnnHandle_t handle, hipdnnFusionPlanDescriptor_t fusePlanDesc,
    size_t *workSpaceSize, hipdnnConvolutionFwdAlgo_t algo) {
    CHECK_MIO(miopenFusionPlanGetWorkSpaceSize(
        (miopenHandle_t)handle, (miopenFusionPlanDescriptor_t)fusePlanDesc,
        workSpaceSize, (miopenConvFwdAlgorithm_t)algo));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnFusionPlanConvolutionGetAlgo(
    hipdnnFusionPlanDescriptor_t fusePlanDesc, const int requestAlgoCount,
    int* returnedAlgoCount, hipdnnConvolutionFwdAlgo_t* returnedAlgos) {

    miopenConvFwdAlgorithm_t mi_returnedAlgos;
    CHECK_MIO(miopenFusionPlanConvolutionGetAlgo(
        (miopenFusionPlanDescriptor_t)fusePlanDesc, requestAlgoCount,
        returnedAlgoCount, &mi_returnedAlgos));
    miopenTohipConvolutionFwdAlgo(mi_returnedAlgos,returnedAlgos);

    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCreateOpConvForward(
    hipdnnFusionPlanDescriptor_t fusePlanDesc,
    hipdnnFusionOpDescriptor_t *convOp, hipdnnConvolutionDescriptor_t convDesc,
    const hipdnnTensorDescriptor_t wDesc) {

    hipdnnConvolutionDescriptor_t convDesc_cast =
                                    ((structConvDesc_t*)(convDesc))->descriptor;
    CHECK_MIO(
        miopenCreateOpConvForward((miopenFusionPlanDescriptor_t)fusePlanDesc,
                                  (miopenFusionOpDescriptor_t *)convOp,
                                  (miopenConvolutionDescriptor_t)convDesc_cast,
                                  (miopenTensorDescriptor_t)wDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCreateOpBiasForward(
    hipdnnFusionPlanDescriptor_t fusePlanDesc,
    hipdnnFusionOpDescriptor_t *biasOp, const hipdnnTensorDescriptor_t bDesc) {
    CHECK_MIO(miopenCreateOpBiasForward(
        (miopenFusionPlanDescriptor_t)fusePlanDesc,
        (miopenFusionOpDescriptor_t *)biasOp, (miopenTensorDescriptor_t)bDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t
hipdnnCreateOpActivationForward(hipdnnFusionPlanDescriptor_t fusePlanDesc,
                                hipdnnFusionOpDescriptor_t *activOp,
                                hipdnnActivationMode_t mode) {
    miopenActivationMode_t mi_mode;
    hipTomiopenActivationMode(mode, &mi_mode);

    CHECK_MIO(miopenCreateOpActivationForward(
        (miopenFusionPlanDescriptor_t)fusePlanDesc,
        (miopenFusionOpDescriptor_t *)activOp, mi_mode));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCreateOpBatchNormInference(
    hipdnnFusionPlanDescriptor_t fusePlanDesc, hipdnnFusionOpDescriptor_t *bnOp,
    const hipdnnBatchNormMode_t bn_mode,
    const hipdnnTensorDescriptor_t bnScaleBiasMeanVarDesc) {

    miopenBatchNormMode_t mi_bn_mode;
    hipTomiopenBatchNormMode(bn_mode, &mi_bn_mode);
    CHECK_MIO(miopenCreateOpBatchNormInference(
        (miopenFusionPlanDescriptor_t)fusePlanDesc,
        (miopenFusionOpDescriptor_t *)bnOp, mi_bn_mode,
        (miopenTensorDescriptor_t)bnScaleBiasMeanVarDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCompileFusionPlan(
    hipdnnHandle_t handle, hipdnnFusionPlanDescriptor_t fusePlanDesc) {
    CHECK_MIO(miopenCompileFusionPlan(
        (miopenHandle_t)handle, (miopenFusionPlanDescriptor_t)fusePlanDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnCreateOperatorArgs(hipdnnOperatorArgs_t *args) {
    CHECK_MIO(miopenCreateOperatorArgs((miopenOperatorArgs_t *)args));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetOpArgsConvForward(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t convOp,
    const void *alpha, const void *beta, const void *w) {
    CHECK_MIO(miopenSetOpArgsConvForward((miopenOperatorArgs_t)args,
                                         (miopenFusionOpDescriptor_t)convOp,
                                         alpha, beta, w));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetOpArgsBiasForward(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t biasOp,
    const void *alpha, const void *beta, const void *bias) {
    CHECK_MIO(miopenSetOpArgsBiasForward((miopenOperatorArgs_t)args,
                                         (miopenFusionOpDescriptor_t)biasOp,
                                         alpha, beta, bias));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetOpArgsActivForward(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t activOp,
    const void *alpha, const void *beta, double activAlpha, double activBeta,
    double activGamma) {
    CHECK_MIO(miopenSetOpArgsActivForward(
        (miopenOperatorArgs_t)args, (miopenFusionOpDescriptor_t)activOp, alpha,
        beta, activAlpha, activBeta, activGamma));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnSetOpArgsBatchNormInference(
    hipdnnOperatorArgs_t args, const hipdnnFusionOpDescriptor_t bnOp,
    const void *alpha, const void *beta, const void *bnScale,
    const void *bnBias, const void *estimatedMean,
    const void *estimatedVariance, double epsilon) {
    CHECK_MIO(miopenSetOpArgsBatchNormInference(
        (miopenOperatorArgs_t)args, (miopenFusionOpDescriptor_t)bnOp, alpha,
        beta, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnExecuteFusionPlan(
    const hipdnnHandle_t handle,
    const hipdnnFusionPlanDescriptor_t fusePlanDesc,
    const hipdnnTensorDescriptor_t inputDesc, const void *input,
    const hipdnnTensorDescriptor_t outputDesc, void *output,
    hipdnnOperatorArgs_t args) {
    CHECK_MIO(miopenExecuteFusionPlan(
        (miopenHandle_t)handle, (miopenFusionPlanDescriptor_t)fusePlanDesc,
        (miopenTensorDescriptor_t)inputDesc, input,
        (miopenTensorDescriptor_t)outputDesc, output,
        (miopenOperatorArgs_t)args));
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnDestroyOperatorArgs(hipdnnOperatorArgs_t args) {
    CHECK_MIO(miopenDestroyOperatorArgs((miopenOperatorArgs_t) args) );
    return HIPDNN_STATUS_SUCCESS;
}

//------------------------------------------------------------------------------

hipdnnStatus_t hipdnnDestroyFusionPlan(
    hipdnnFusionPlanDescriptor_t fusePlanDesc) {
    CHECK_MIO(
        miopenDestroyFusionPlan((miopenFusionPlanDescriptor_t)fusePlanDesc));
    return HIPDNN_STATUS_SUCCESS;
}

//==============================================================================

#endif //WITH_HIP
