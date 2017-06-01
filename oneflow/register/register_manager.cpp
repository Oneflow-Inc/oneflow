#include "register/register_manager.h"

namespace oneflow {

void RegstMgr::NewRegsts(const RegstDescProto& regst_desc_proto,
               std::function<void(Regst*)> OneRegstDone) {
  // One RegstDesc means Multi Regst
  // All Regst has a shared_ptr point to the same RtRegstDesc obj
  // Call OneRegstDone for each regst
  std::shared_ptr<RtRegstDesc> runtime_regst_desc(new RtRegstDesc(regst_desc_proto));
  for (size_t i = 0; i < regst_desc_proto.register_num(); ++i) {
    Regst* regst = new Regst;
    regst->regst_desc_ = runtime_regst_desc;
    regst->regst_id_ = i;
    for (const auto& pair : regst_desc_proto.lbn2shape()) {
      Shape* shape_ptr = runtime_regst_desc->GetShapePtrFromLbn(pair.first);
      regst->lbn2blob_.emplace(pair.first, 
          of_make_unique<Blob>(new char[shape_ptr->elem_cnt()], shape_ptr));
    }
    OneRegstDone(regst);
  }
}

}
