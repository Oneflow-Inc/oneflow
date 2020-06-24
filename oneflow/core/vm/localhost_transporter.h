#ifndef ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_
#define ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_

#include "oneflow/core/vm/transporter.h"

namespace oneflow {
namespace vm {

class LocalhostTransporter final : public Transporter {
 public:
  LocalhostTransporter() = default;
  ~LocalhostTransporter() override = default;

  void MakeSendTransportRequest(
      const TransportDataToken& data_token, const char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2SendRequest* transport_key2send_request) const override;

  void MakeReceiveTransportRequest(
      const TransportDataToken& data_token, char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2ReceiveRequest* transport_key2send_request) const override;

  void Transport(TransportKey2SendRequest* transport_key2send_request) const override;
  void Transport(TransportKey2ReceiveRequest* transport_key2receive_request) const override;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_
