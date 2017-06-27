#ifndef ONEFLOW_CORE_REGISTER_REGISTER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/runtime_register_desc.h"

namespace oneflow {

class Regst final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Regst);
  ~Regst() {
    deleter_();
  }

  // Getters
  int64_t piece_id() const { return piece_id_; }
  int64_t model_version_id() const { return model_version_id_; }
  int64_t regst_desc_id() const { return regst_desc_->regst_desc_id(); }
  int64_t producer_actor_id() const {
    return regst_desc_->producer_actor_id();
  }
  const std::vector<int64_t>& subscribers_actor_id() const {
    return regst_desc_->subscribers_actor_id();
  }
  void ForEachLbn(std::function<void(const std::string&)>);

  // Setters
  void set_piece_id(int64_t val) { piece_id_ = val; }
  void set_model_version_id(int64_t val) { model_version_id_ = val; }
  Blob* GetBlobPtrFromLbn(const std::string& lbn);
    
 private:
  friend class RegstMgr;
  Regst();

  int64_t piece_id_;
  int64_t model_version_id_;

  std::shared_ptr<const RtRegstDesc> regst_desc_;
  std::function<void()> deleter_;
  HashMap<std::string, std::unique_ptr<Blob>> lbn2blob_;
  std::unique_ptr<Blob> baled_blob_;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_REGISTER_REGISTER_H_
