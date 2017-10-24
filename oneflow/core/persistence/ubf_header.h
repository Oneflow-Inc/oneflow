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
  UbfHeader(uint32_t ubf_item_num, const std::vector<uint32_t>& dim_array);

  bool ValidateMagicCode() const { return magic_code_ == 0xfeed; }
  //  getter
  uint64_t ubf_item_num() const { return ubf_item_num_; }
  uint32_t dim_array_size() const { return dim_array_size_; }

 private:
  const uint16_t magic_code_ = 0xfeed;
  uint16_t dim_array_size_ = 0;  //  effective length of dim_array
  uint32_t dim_array_[15];       //  tensor shape
  uint64_t ubf_item_num_ = 0;    //  how many items after header
};

PersistentOutStream& operator<<(PersistentOutStream& out,
                                const UbfHeader& data);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_UBF_HEADER_H_
