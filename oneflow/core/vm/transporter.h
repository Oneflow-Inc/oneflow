#ifndef ONEFLOW_CORE_VM_TRANSPORTER_H_
#define ONEFLOW_CORE_VM_TRANSPORTER_H_

#include "oneflow/core/vm/transport_request.msg.h"

namespace oneflow {
namespace vm {

class Transporter {
 public:
  Transporter(const Transporter&) = delete;
  Transporter(Transporter&&) = delete;
  virtual ~Transporter() = default;

  virtual void MakeSendTransportRequest(
      const TransportDataToken& data_token, const char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2SendRequest* transport_key2send_request) const = 0;

  virtual void MakeReceiveTransportRequest(
      const TransportDataToken& data_token, char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2ReceiveRequest* transport_key2send_request) const = 0;

  virtual void Transport(TransportKey2SendRequest* transport_key2send_request) const = 0;
  virtual void Transport(TransportKey2ReceiveRequest* transport_key2receive_request) const = 0;

 protected:
  Transporter() = default;
};

}  // namespace vm

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TRANSPORTER_H_
