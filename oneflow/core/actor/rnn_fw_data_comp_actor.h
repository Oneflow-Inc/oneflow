#ifndef ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RnnFwDataCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnFwDataCompActor);
  RnnFwDataCompActor() = default;
  ~RnnFwDataCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  struct DataLoadBuf {
    DataLoadBuf()
        : row_num(0),
          col_num(0),
          col_id(0),
          max_real_col_num(0),
          piece(),
          data_ids() {}

    int64_t row_num;
    int64_t col_num;
    int64_t col_id;
    int64_t max_real_col_num;
    std::vector<int64_t> piece;
    std::vector<std::string> data_ids;
  };

  int WaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;
  int HandlerWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  void AsyncSendMsgToModelAndModelTmpProducer();

  CudaStreamHandle cuda_handle_;
  int64_t expected_model_version_id_;
  int64_t in_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  Regst* model_regst_;
  Regst* model_tmp_regst_;
  std::queue<Regst*> in_;
  HashMap<int64_t, Regst*> readable_regst_;
  KernelCtx kernel_ctx_;
  DataLoadBuf data_load_buf_;
  PieceStatus expected_piece_status_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
