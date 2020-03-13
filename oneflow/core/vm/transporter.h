#ifndef ONEFLOW_CORE_VM_TRANSPORTER_H_
#define ONEFLOW_CORE_VM_TRANSPORTER_H_

#include "oneflow/core/vm/transport_request.msg.h"

namespace oneflow {

class Transporter {
 public:
  virtual void MakeReadTransportRequest(
      uint64_t data_token, const char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2ReadRequest* transport_key2read_request) const = 0;

  virtual void MakeWriteTransportRequest(
      uint64_t data_token, char* data_ptr, size_t data_size, std::atomic<int64_t>* incomplete_cnt,
      TransportKey2WriteRequest* transport_key2read_request) const = 0;

  virtual void Transport(TransportKey2ReadRequest* transport_key2read_request) const = 0;
  virtual void Transport(TransportKey2WriteRequest* transport_key2write_request) const = 0;

 protected:
  Transporter() = default;
  virtual ~Transporter() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_TRANSPORTER_H_
