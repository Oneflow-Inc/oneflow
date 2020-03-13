#ifndef ONEFLOW_CORE_VM_TRANSPORT_REQUEST_H_
#define ONEFLOW_CORE_VM_TRANSPORT_REQUEST_H_

#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/object_msg.h"

namespace oneflow {

// clang-format off

FLAT_MSG_BEGIN(TransportMirroredDataToken);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, logical_token);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, parallel_id);
FLAT_MSG_END(TransportMirroredDataToken);

FLAT_MSG_BEGIN(TransportDataToken);
  FLAT_MSG_DEFINE_ONEOF(type,
    FLAT_MSG_ONEOF_FIELD(uint64_t, token)
    FLAT_MSG_ONEOF_FIELD(TransportMirroredDataToken, mirrored_token));
FLAT_MSG_END(TransportDataToken);

FLAT_MSG_BEGIN(TransportKey);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(TransportDataToken, data_token);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, data_offset);

  // methods
  PUBLIC FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
FLAT_MSG_END(TransportKey);

FLAT_MSG_BEGIN(TransportSize);
  // fields
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, total_data_size);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, current_transport_capacity);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, current_valid_size);

  // methods
  PUBLIC FLAT_MSG_DEFINE_COMPARE_OPERATORS_BY_MEMCMP();
FLAT_MSG_END(TransportSize);

FLAT_MSG_BEGIN(TransportHeader);
  FLAT_MSG_DEFINE_OPTIONAL(TransportKey, key);
  FLAT_MSG_DEFINE_OPTIONAL(TransportSize, size);
FLAT_MSG_END(TransportHeader);

enum TransportRequestType { kReadTransportRequestType = 0, kWriteTransportRequestType };

template<TransportRequestType request_type>
struct TransportRequestDataType {};

template<>
struct TransportRequestDataType<kReadTransportRequestType> {
  using type = const char;
};

template<>
struct TransportRequestDataType<kWriteTransportRequestType> {
  using type = char;
};

template<TransportRequestType request_type>
OBJECT_MSG_BEGIN(TransportRequest);
  // fields  
  OBJECT_MSG_DEFINE_RAW_PTR(typename TransportRequestDataType<request_type>::type, data_ptr);
  OBJECT_MSG_DEFINE_RAW_PTR(std::atomic<int64_t>, incomplete_cnt);
  OBJECT_MSG_DEFINE_FLAT_MSG(TransportSize, size);

  // links
  OBJECT_MSG_DEFINE_MAP_FLAT_MSG_KEY(TransportKey, transport_key);
OBJECT_MSG_END(TransportRequest);

using ReadTransportRequest = TransportRequest<kReadTransportRequestType>;
using WriteTransportRequest = TransportRequest<kWriteTransportRequestType>;

template<TransportRequestType request_type>
using TransportKey2Request = OBJECT_MSG_MAP(TransportRequest<request_type>, transport_key);

using TransportKey2ReadRequest = TransportKey2Request<kReadTransportRequestType>;
using TransportKey2WriteRequest = TransportKey2Request<kWriteTransportRequestType>;

// clang-format on

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TRANSPORT_REQUEST_H_
