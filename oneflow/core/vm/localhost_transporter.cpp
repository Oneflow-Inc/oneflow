#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(LocalhostTransportRequestHub);
  // fields
  OBJECT_MSG_DEFINE_RAW(std::mutex, mutex);
  
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(ReadTransportRequest, transport_key, transport_key2read_request);
  OBJECT_MSG_DEFINE_MAP_HEAD(WriteTransportRequest, transport_key, transport_key2write_request);
OBJECT_MSG_END(LocalhostTransportRequestHub);
// clang-format on

LocalhostTransportRequestHub* GetTransportRequestHub() {
  static auto hub = ObjectMsgPtr<LocalhostTransportRequestHub>::New();
  return hub.Mutable();
}

template<TransportRequestType request_type>
void MakeTransportRequest(uint64_t data_token,
                          typename TransportRequestDataPointer<request_type>::type data_ptr,
                          size_t data_size, volatile std::atomic<int64_t>* incomplete_cnt,
                          TransportKey2Request<request_type>* transport_key2request) {
  auto read_request = ObjectMsgPtr<TransportRequest<request_type>>::New();
  read_request->set_data_ptr(data_ptr);
  read_request->set_incomplete_cnt(incomplete_cnt);
  read_request->mutable_size()->set_total_data_size(data_size);
  read_request->mutable_size()->set_current_transport_capacity(GetMaxVal<int64_t>());
  read_request->mutable_size()->set_current_valid_size(data_size);
  read_request->mutable_transport_key()->set_data_token(data_token);
  read_request->mutable_transport_key()->set_data_offset(0);
  CHECK(transport_key2request->Insert(read_request.Mutable()).second);
}

}  // namespace

void LocalhostTransporter::MakeReadTransportRequest(
    uint64_t data_token, const char* data_ptr, size_t data_size,
    volatile std::atomic<int64_t>* incomplete_cnt,
    TransportKey2ReadRequest* transport_key2read_request) const {
  MakeTransportRequest<kReadTransportRequestType>(data_token, data_ptr, data_size, incomplete_cnt,
                                                  transport_key2read_request);
}

void LocalhostTransporter::MakeWriteTransportRequest(
    uint64_t data_token, char* data_ptr, size_t data_size,
    volatile std::atomic<int64_t>* incomplete_cnt,
    TransportKey2WriteRequest* transport_key2read_request) const {
  MakeTransportRequest<kWriteTransportRequestType>(data_token, data_ptr, data_size, incomplete_cnt,
                                                   transport_key2read_request);
}

void LocalhostTransporter::Transport(TransportKey2ReadRequest* transport_key2read_request) const {
  TODO();
}

void LocalhostTransporter::Transport(TransportKey2WriteRequest* transport_key2write_request) const {
  TODO();
}

}  // namespace oneflow
