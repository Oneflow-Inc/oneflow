#include "oneflow/core/distributed_runtime/grpc_tensor_coding.h"
#include "oneflow/core/distributed_runtime/worker.pb.h"
#include "oneflow/core/proto/tensor.pb.h"
#include "oneflow/core/proto/tensor_shape.pb.h"
#include <glog/logging.h>
#include "tensorflow/core/lib/io/proto_encode_helper.h"

#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"

#include "oneflow/core/distributed_runtime/tensor_shape.h"

namespace oneflow {

namespace grpc {

void EncodeRecvTensorResponseToByteBuffer(const ReadDataResponse& proto,
                                          ::grpc::ByteBuffer* result) {

}

static void do_nothing(void* raw) {}

static int VarLengthEncodingSize(uint32_t tag, size_t bytes) {
  return ::tensorflow::core::VarintLength(tag << 3)
    + ::tensorflow::core::VarintLength(bytes) + bytes;
}

static int SkeletonEncodingSizeUpperBound(const Tensor& val) {
  static const int kVarintMax64 = 10;
  const int ndims = val.shape().dims();
  return (2 * kVarintMax64) + (ndims * (4 * kVarintMax64));
}

static void EncodeSkeleton(const Tensor& val, ::tensorflow::io::ProtoEncodeHelper* e) {
  //e->WriteUint64(TensorProto::kDtypeFieldNumber, val.dtype());
  const int ndims = val.shape().dims();
  int32_t tensor_shape_bytes = 0;
  for(int d = 0; d < ndims; d++) {
    int64_t dim_size = val.shape().dim_size(d);
    tensor_shape_bytes += 2 + 1 + ::tensorflow::core::VarintLength(dim_size);
  }

  if(tensor_shape_bytes > 0) {
    e->WriteVarlengthBeginning(TensorProto::kTensorShapeFieldNumber, tensor_shape_bytes);
    for(int d = 0; d < ndims; d++) {
      int64_t dim_size = val.shape().dim_size(d);
      int64_t dim_varlen = 1 + ::tensorflow::core::VarintLength(dim_size);
      e->WriteVarlengthBeginning(TensorShapeProto::kDimFieldNumber, dim_varlen);
      e->WriteUint64(TensorShapeProto_Dim::kSizeFieldNumber, dim_size);
    }
  }
}  // EncodeSkeleton

void EncodeTensorToByteBuffer(bool is_dead, const Tensor& val, ::grpc::ByteBuffer* result) {
  const int kLargeTensorBytes = 1024;
  ReadDataResponse response;

  std::vector<char> skeleton(SkeletonEncodingSizeUpperBound(val));
  ::tensorflow::io::ProtoEncodeHelper e_skeleton(skeleton.data(), skeleton.size());
  EncodeSkeleton(val, &e_skeleton);

  std::string tdata = val.tensor_data();
  // Tensr internal structure
  uint32_t overall_tensor_proto_bytesize = 
    (e_skeleton.size() +
     VarLengthEncodingSize(TensorProto::kTensorContentFieldNumber, tdata.size()));

  std::string header;
  response.AppendToString(&header);
  // header size + Tensor size 
  size_t expected_size =
    (header.size() +
     VarLengthEncodingSize(ReadDataResponse::kTensorFieldNumber,overall_tensor_proto_bytesize));
  // only tensor
  bool is_large = (tdata.size() > kLargeTensorBytes);
  // encode_size is header size;
  size_t encode_size = expected_size - tdata.size();
  std::vector<char> space(encode_size);
  ::tensorflow::io::ProtoEncodeHelper e(space.data(), space.size());

  e.WriteRawBytes(header);
  e.WriteVarlengthBeginning(ReadDataResponse::kTensorFieldNumber,
                            overall_tensor_proto_bytesize);

  // e.WriteRawBytes(e_skeleton.data(), e_skeleton.size());
  e.WriteVarlengthBeginning(TensorProto::kTensorContentFieldNumber,
                            tdata.size());

  ::grpc::Slice slices[3];
  int num_slices = 0;
  {
    size_t slice_len = e.size() + (is_large ? 0 : tdata.size());
    gpr_slice s0 = gpr_slice_malloc(slice_len);
    // memcpy(static_cast<void*>(s0), e.data(), e.size());
    slices[0] = ::grpc::Slice(s0, ::grpc::Slice::STEAL_REF);
    num_slices += 1;
  }
  if(is_large) {
    gpr_slice s1 =
      gpr_slice_new(const_cast<void*>(static_cast<const void*>(tdata.data())), 
          tdata.size(), do_nothing);
    slices[1] = ::grpc::Slice(s1, ::grpc::Slice::STEAL_REF);

    // gpr_slice s2 =
    //   gpr_slice_new(const_cast<>);
  }

  size_t total_bytes = 0;
  for(int i = 0; i < num_slices; ++i) {
    total_bytes += slices[i].size();
  }

  *result = ::grpc::ByteBuffer(&slices[0], num_slices);
}  // EncodeTensorToByteBuffer

}  // namespace grpc

}  // namespace oneflow
