#ifndef ONEFLOW_CORE_PERSISTENCE_UBF_HEADER_H_
#define ONEFLOW_CORE_PERSISTENCE_UBF_HEADER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

//	binary format
//	.-------------------------.
//	| UbfHeader | UbfItem ... |
//	'-------------------------'

//  united binary format header
class UbfHeader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfHeader);
  UbfHeader() = default;
  ~UbfHeader() = default;
  UbfHeader(const std::string& type, uint32_t ubf_item_num,
            const std::vector<uint32_t>& dim_array);

  bool ValidateMagicCode() const { return magic_code_ == 0xfeed; }
  //  add all bytes one by one
  uint32_t ComputeCheckSum() const;
  void UpdateCheckSum();

  //  getter
  uint64_t ubf_item_num() const { return ubf_item_num_; }

 private:
  const uint16_t magic_code_ = 0xfeed;
  const uint16_t version_ = 0;
  uint32_t check_sum_;           // check header
  char type_[16];                //  "feature" or "label"
  uint32_t dim_array_size_ = 0;  //  effective length of dim_array
  uint32_t dim_array_[15];       //  tensor shape
  uint64_t ubf_item_num_ = 0;    //  how many items after header
};

PersistentOutStream& operator<<(PersistentOutStream& out,
                                const UbfHeader& data);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_UBF_HEADER_H_
