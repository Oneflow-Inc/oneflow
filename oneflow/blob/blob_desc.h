#ifndef ONEFLOW_BLOB_BLOB_DESC_H_
#define ONEFLOW_BLOB_BLOB_DESC_H_

#include "common/util.h"
#include "common/shape.h"

namespace oneflow {

class BlobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobDesc);
  BlobDesc() = default;
  ~BlobDesc() = default;
  
  // Getters
  const std::string& pbn() const { return pbn_; }
  const std::string& lbn() const { return lbn_; }
  const Shape& shape() const { return shape_; }

  // Setters
  std::string& mut_pbn() { return pbn_; }
  std::string& mut_lbn() { return lbn_; }
  Shape& mut_shape() { return shape_; }

 private:
  std::string pbn_;
  std::string lbn_;
  Shape shape_;
};

} // namespace oneflow

#endif // ONEFLOW_BLOB_BLOB_DESC_H_
