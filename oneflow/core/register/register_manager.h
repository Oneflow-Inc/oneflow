#ifndef ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
#define ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_

#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/memory/memory_allocator.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class RegstMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegstMgr);
  RegstMgr() = delete;
  ~RegstMgr() = default;

  void NewRegsts(const RegstDescProto& regst_desc_proto, std::function<void(Regst*)> OneRegstDone);

 private:
  friend class Global<RegstMgr>;

  explicit RegstMgr(const Plan& plan);
  explicit RegstMgr(const std::list<const RegstDescProto*>& regst_protos);
  void InitFromRegstProtoList(const std::list<const RegstDescProto*>& regst_protos);
  void InitPbBlobIfNeed(Blob* blob_ptr) const;
  template<typename T>
  void InitPbBlob(Blob* blob_ptr) const;
  void NewBlobsInOneRegst(const std::vector<LbiBlobDescPair>& lbis, Regst*, const RtRegstDesc*,
                          char* main_mem_ptr);

  HashMap<int64_t, std::unique_ptr<const RtRegstDesc>> regst_desc_id2rt_regst_desc_;
  HashMap<int64_t, char*> regst_desc_id2main_mem_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_REGISTER_MANAGER_H_
