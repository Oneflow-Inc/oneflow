#ifndef ONEFLOW_CORE_ACTOR_RNN_SOURCE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RNN_SOURCE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RnnSourceCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnSourceCompActor);
  RnnSourceCompActor() = default;
  ~RnnSourceCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto&) override;

  int HandlerWaitToStart(const ActorMsg&);
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override { return IsReadReady(); }
  void AsyncReturnAllReadableRegst() override {}

  struct DataLoadBuf {
    DataLoadBuf()
        : is_initiated(false),
          piece_id(0),
          row_num(0),
          col_num(0),
          col_id(0),
          max_real_col_num(0),
          piece(),
          data_ids(),
          offsets() {}

    bool is_initiated;
    int64_t piece_id;
    int64_t row_num;
    int64_t col_num;
    int64_t col_id;
    int64_t max_real_col_num;
    std::vector<int64_t> piece;
    std::vector<std::string> data_ids;
    std::vector<BlobDesc::OffSetType> offsets;
  };

  DataLoadBuf data_load_buf_;
  bool is_eof_;
  bool is_final_piece_done_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RNN_SOURCE_COMPUTE_ACTOR_H_
