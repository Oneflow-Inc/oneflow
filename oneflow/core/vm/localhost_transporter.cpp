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
void MakeTransportRequest(const TransportDataToken& data_token,
                          typename TransportRequestDataType<request_type>::type* data_ptr,
                          size_t data_size, std::atomic<int64_t>* incomplete_cnt,
                          TransportKey2Request<request_type>* transport_key2request) {
  auto read_request = ObjectMsgPtr<TransportRequest<request_type>>::New();
  read_request->set_data_ptr(data_ptr);
  read_request->set_incomplete_cnt(incomplete_cnt);
  read_request->mutable_size()->set_total_data_size(data_size);
  read_request->mutable_size()->set_current_transport_capacity(GetMaxVal<int64_t>());
  read_request->mutable_size()->set_current_valid_size(data_size);
  read_request->mutable_transport_key()->mutable_data_token()->CopyFrom(data_token);
  read_request->mutable_transport_key()->set_data_offset(0);
  CHECK(transport_key2request->Insert(read_request.Mutable()).second);
}

void CopyAndDecreaseIncompleteCnt(WriteTransportRequest* write_request,
                                  ReadTransportRequest* read_request) {
  CHECK(write_request->size() == read_request->size());
  CHECK(write_request->transport_key() == read_request->transport_key());
  std::memcpy(write_request->mut_data_ptr(), &read_request->data_ptr(),
              read_request->size().current_valid_size());
  CHECK_GT(write_request->incomplete_cnt(), 0);
  CHECK_GT(read_request->incomplete_cnt(), 0);
  --*write_request->mut_incomplete_cnt();
  --*read_request->mut_incomplete_cnt();
}

}  // namespace

void LocalhostTransporter::MakeReadTransportRequest(
    const TransportDataToken& data_token, const char* data_ptr, size_t data_size,
    std::atomic<int64_t>* incomplete_cnt,
    TransportKey2ReadRequest* transport_key2read_request) const {
  MakeTransportRequest<kReadTransportRequestType>(data_token, data_ptr, data_size, incomplete_cnt,
                                                  transport_key2read_request);
}

void LocalhostTransporter::MakeWriteTransportRequest(
    const TransportDataToken& data_token, char* data_ptr, size_t data_size,
    std::atomic<int64_t>* incomplete_cnt,
    TransportKey2WriteRequest* transport_key2read_request) const {
  MakeTransportRequest<kWriteTransportRequestType>(data_token, data_ptr, data_size, incomplete_cnt,
                                                   transport_key2read_request);
}

void LocalhostTransporter::Transport(TransportKey2ReadRequest* transport_key2read_request) const {
  auto* hub = GetTransportRequestHub();
  OBJECT_MSG_MAP_FOR_EACH_PTR(transport_key2read_request, read_reqeust) {
    {
      std::unique_lock<std::mutex> lock(*hub->mut_mutex());
      auto* write_request =
          hub->mut_transport_key2write_request()->FindPtr(read_reqeust->transport_key());
      if (write_request == nullptr) {
        CHECK(hub->mut_transport_key2read_request()->Insert(read_reqeust).second);
      } else {
        CopyAndDecreaseIncompleteCnt(write_request, read_reqeust);
        hub->mut_transport_key2write_request()->Erase(write_request);
      }
    }
    transport_key2read_request->Erase(read_reqeust);
  }
}

void LocalhostTransporter::Transport(TransportKey2WriteRequest* transport_key2write_request) const {
  auto* hub = GetTransportRequestHub();
  OBJECT_MSG_MAP_FOR_EACH_PTR(transport_key2write_request, write_request) {
    {
      std::unique_lock<std::mutex> lock(*hub->mut_mutex());
      auto* read_request =
          hub->mut_transport_key2read_request()->FindPtr(write_request->transport_key());
      if (read_request == nullptr) {
        CHECK(hub->mut_transport_key2write_request()->Insert(write_request).second);
      } else {
        CopyAndDecreaseIncompleteCnt(write_request, read_request);
        hub->mut_transport_key2read_request()->Erase(read_request);
      }
    }
    transport_key2write_request->Erase(write_request);
  }
}

}  // namespace oneflow
