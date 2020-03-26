#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {
namespace vm {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(LocalhostTransportRequestHub);
  // fields
  OBJECT_MSG_DEFINE_STRUCT(std::mutex, mutex);
  
  // links
  OBJECT_MSG_DEFINE_MAP_HEAD(SendTransportRequest, transport_key, transport_key2send_request);
  OBJECT_MSG_DEFINE_MAP_HEAD(ReceiveTransportRequest, transport_key, transport_key2receive_request);
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
  auto request = ObjectMsgPtr<TransportRequest<request_type>>::New();
  request->set_data_ptr(data_ptr);
  request->set_incomplete_cnt(incomplete_cnt);
  request->mutable_size()->set_total_data_size(data_size);
  request->mutable_size()->set_current_transport_capacity(GetMaxVal<int64_t>());
  request->mutable_size()->set_current_valid_size(data_size);
  request->mutable_transport_key()->mutable_data_token()->CopyFrom(data_token);
  request->mutable_transport_key()->set_data_offset(0);
  CHECK(transport_key2request->Insert(request.Mutable()).second);
}

void CopyAndDecreaseIncompleteCnt(ReceiveTransportRequest* receive_request,
                                  SendTransportRequest* send_request) {
  CHECK(receive_request->size() == send_request->size());
  CHECK(receive_request->transport_key() == send_request->transport_key());
  std::memcpy(receive_request->mut_data_ptr(), &send_request->data_ptr(),
              send_request->size().current_valid_size());
  CHECK_GT(receive_request->incomplete_cnt(), 0);
  CHECK_GT(send_request->incomplete_cnt(), 0);
  --*receive_request->mut_incomplete_cnt();
  --*send_request->mut_incomplete_cnt();
}

}  // namespace

void LocalhostTransporter::MakeSendTransportRequest(
    const TransportDataToken& data_token, const char* data_ptr, size_t data_size,
    std::atomic<int64_t>* incomplete_cnt,
    TransportKey2SendRequest* transport_key2send_request) const {
  MakeTransportRequest<kSendTransportRequestType>(data_token, data_ptr, data_size, incomplete_cnt,
                                                  transport_key2send_request);
}

void LocalhostTransporter::MakeReceiveTransportRequest(
    const TransportDataToken& data_token, char* data_ptr, size_t data_size,
    std::atomic<int64_t>* incomplete_cnt,
    TransportKey2ReceiveRequest* transport_key2send_request) const {
  MakeTransportRequest<kReceiveTransportRequestType>(data_token, data_ptr, data_size,
                                                     incomplete_cnt, transport_key2send_request);
}

void LocalhostTransporter::Transport(TransportKey2SendRequest* transport_key2send_request) const {
  auto* hub = GetTransportRequestHub();
  OBJECT_MSG_MAP_FOR_EACH(transport_key2send_request, send_request) {
    transport_key2send_request->Erase(send_request.Mutable());
    ObjectMsgPtr<ReceiveTransportRequest> receive_request;
    {
      std::unique_lock<std::mutex> lock(*hub->mut_mutex());
      receive_request =
          hub->mut_transport_key2receive_request()->Find(send_request->transport_key());
      if (receive_request) {
        hub->mut_transport_key2receive_request()->Erase(receive_request.Mutable());
      } else {
        CHECK(hub->mut_transport_key2send_request()->Insert(send_request.Mutable()).second);
      }
    }
    if (receive_request) {
      CopyAndDecreaseIncompleteCnt(receive_request.Mutable(), send_request.Mutable());
    }
  }
}

void LocalhostTransporter::Transport(
    TransportKey2ReceiveRequest* transport_key2receive_request) const {
  auto* hub = GetTransportRequestHub();
  OBJECT_MSG_MAP_FOR_EACH(transport_key2receive_request, receive_request) {
    transport_key2receive_request->Erase(receive_request.Mutable());
    ObjectMsgPtr<SendTransportRequest> send_request;
    {
      std::unique_lock<std::mutex> lock(*hub->mut_mutex());
      send_request = hub->mut_transport_key2send_request()->Find(receive_request->transport_key());
      if (send_request) {
        hub->mut_transport_key2send_request()->Erase(send_request.Mutable());
      } else {
        CHECK(hub->mut_transport_key2receive_request()->Insert(receive_request.Mutable()).second);
      }
    }
    if (send_request) {
      CopyAndDecreaseIncompleteCnt(receive_request.Mutable(), send_request.Mutable());
    }
  }
}

}  // namespace vm
}  // namespace oneflow
