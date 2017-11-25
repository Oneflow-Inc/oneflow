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

  struct Material4Act {
    enum class RnnKernelType {
      kDataLoader = 0,
      kNormal,
      kRnnCellWithInitial,
      kRnnCellWithoutInitial
    };
    Material4Act() : rnn_kernel_type(RnnKernelType::kDataLoader),
                     regst_id2regst() { }

    RnnKernelType rnn_kernel_type;
    std::map<int64_t, Regst*> regst_id2regst;
  };

  int WaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;
  int HandlerWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override;
  void Act() override;
  void AsyncSendMsgToModelAndModelTmpProducer();
  bool ModelSatisfySSP(int64_t piece_id, int64_t model_version_id) const;
  void set_material4act(Material4Act::RnnKernelType type,
                        Regst* in_regst, Regst* model_regst, 
                        Regst* initial_regst, Regst* out_regst);
  void UpdtInAndModelStatesOfRnnCell();

  CudaStreamHandle cuda_handle_;
  int64_t expected_model_version_id_;

  int64_t in_regst_desc_id_;
  std::map<int64_t, std::queue<Regst*>> pid2in_regsts_; // <piece_id, in_regst>
  // must be increasing in iteration, so using std::map instead of HashMap

  int64_t initial_hidden_regst_desc_id_;
  std::queue<Regst*> initial_hidden_regsts_;

  int64_t model_tmp_regst_desc_id_;
  Regst* model_tmp_regst_;

  int64_t model_regst_desc_id_;
  Regst* latest_model_regst_;
  HashMap<int64_t, Regst*> pid2model_regst_;
  HashMap<Regst*, int64_t> model_regst2cnt_;  // <model_regst, number of piece using this model>

  int64_t out_regst_desc_id_;
  HashMap<int64_t, Regst*> pid2out_regst_;

  HashMap<int64_t, Regst*> readable_regst_;
  KernelCtx kernel_ctx_;

  DataLoadBuf data_load_buf_;
  PieceStatus expected_piece_status_;
  Material4Act material4act_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_FW_DATA_COMP_ACTOR_H_
