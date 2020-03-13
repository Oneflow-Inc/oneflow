#ifndef ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_
#define ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_

#include "oneflow/core/vm/transporter.h"

namespace oneflow {

class LocalhostTransporter final : public Transporter {
 public:
  LocalhostTransporter() = default;
  ~LocalhostTransporter() override = default;

  void MakeReadTransportRequest(
      uint64_t data_token, const char* data_ptr, size_t data_size,
      std::atomic<int64_t>* incomplete_cnt,
      TransportKey2ReadRequest* transport_key2read_request) const override;

  void MakeWriteTransportRequest(
      uint64_t data_token, char* data_ptr, size_t data_size, std::atomic<int64_t>* incomplete_cnt,
      TransportKey2WriteRequest* transport_key2read_request) const override;

  void Transport(TransportKey2ReadRequest* transport_key2read_request) const override;
  void Transport(TransportKey2WriteRequest* transport_key2write_request) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_LOCALHOST_TRANSPORTER_H_
