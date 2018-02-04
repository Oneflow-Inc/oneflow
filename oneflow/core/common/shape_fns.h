#ifndef ONEFLOW_CORE_COMMON_SHAPE_FNS_H_
#define ONEFLOW_CORE_COMMON_SHAPE_FNS_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

// GetWindowedOutputSize(): Given an input tensor, kernel, stride and padding
// type, the function computes the output and padding dimensions.
//
// For example, ignoring batches or multiple features, a 1D convolution
// takes as input a 1D tensor of shape (H), and convolves it with a filter of
// shape (K).
//
// It also takes in a few additional parameters:
//
// Stride (S): the stride with which we apply the filters. This is the offset
// between locations where we apply the filters. A larger stride
// means that the output will be spatially smaller.
//
// Padding (P): the padding we apply to the input tensor along each
// dimension. This is usually used to make sure that the spatial dimensions
// do not shrink when we progress with convolutions. Two types of padding are
// often used:
//   SAME: the pad value is computed so that the output will have size H/S.
//   VALID: no padding is carried out.
// The padded area is zero-filled.
//
// The output dimensions for convolution and many other operations, when given
// all the parameters above, are as follows:
// - When Padding = SAME: the output size is (H'), where
//     H' = ceil(float(H) / float(S))
//   where ceil is the ceiling function. The number of padded cells
//   is computed as:
//     Pc = ((H' - 1) * S + K - H) / 2
//   When the stride is 1, the expression simplifies to
//     H' = H, Pc = (K-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     H' = ceil(float(H - K + 1) / float(S))
//   and the number of padded cells is always zero.
//   When the stride is 1, the expression simplifies to
//     H' = H-K+1.
//
// For convolution, mathematically, the output value at location (r')
// is the inner product of two vectors: the chunk of input at
//    ((r'*S-Pr) : (r'*S-Pr+K)),
// and the filter.
//
// For 2D and 3D convolutions, the spatial dimensions are orthogonal, so the
// size and padding of each spatial dimension can be computed by calling
// GetWindowedOutputSize separately for each dimension.
//
void GetWindowedOutputSize(int64_t input_size, int64_t filter_size,
                           int64_t stride, const std::string& padding_type,
                           int64_t* output_size, int64_t* padding_size);

// The V2 version computes the same outputs with arbitrary dilation_rate.
// The output dimensions are computed as follows:
// - When adding dilation_rate (D), we compute an effective filter size (K'):
//     K' = (K - 1) * D + 1
// - When Padding = SAME: the output size is (H'), where
//     H' = ceil(float(H) / float(S))
//   where ceil is the ceiling function. The number of padded cells
//   is computed as:
//     Pc = ((H' - 1) * S + K' - H) / 2
//   When the stride is 1, the expression simplifies to
//     H' = H, Pc = (K'-1)/2.
//   This is where SAME comes from - the output has the same size as the input
//   has.
//
// - When Padding = VALID: the output size is computed as
//     H' = ceil(float(H - K' + 1) / float(S))
//   and the number of padded cells is always zero.
//   When the stride is 1, the expression simplifies to
//     H' = H-K'+1.
//
// TODO(b/67112639): Merge V2 versions and the original versions eventually.
void GetWindowedOutputSizeV2(int64_t input_size, int64_t filter_size,
                             int64_t dilation_rate, int64_t stride,
                             const std::string& padding_type,
                             int64_t* output_size, int64_t* padding_size);

// Returns the same output dimensions as in GetWindowedOutputSize, but returns
// verbose padding dimensions (before/after). Any excess padding
// (caused by an odd padding size value) is added to the 'padding_after'
// dimension.
void GetWindowedOutputSizeVerbose(int64_t input_size, int64_t filter_size,
                                  int64_t stride,
                                  const std::string& padding_type,
                                  int64_t* output_size, int64_t* padding_before,
                                  int64_t* padding_after);

// The V2 version computes the same outputs with arbitrary dilation_rate. For
// detailed equations, refer to the comments for GetWindowedOutputSizeV2().
void GetWindowedOutputSizeVerboseV2(int64_t input_size, int64_t filter_size,
                                    int64_t dilation_rate, int64_t stride,
                                    const std::string& padding_type,
                                    int64_t* output_size,
                                    int64_t* padding_before,
                                    int64_t* padding_after);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_FNS_H_
