#ifndef ONEFLOW_CORE_ACTOR_OF_SERVING_H_
#define ONEFLOW_CORE_ACTOR_OF_SERVING_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/sink_compute_actor.h"
#include "oneflow/core/rpc_service/rpc_server.h"
#include "oneflow/core/rpc_service/connection.h"

namespace oneflow {

class OFServing final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFServing);

  void SendMsg(const std::pair<int64_t, std::string>& data) {
    int conn_id = data.first;
    if (conn_id == -1) { return; }

    rpc_service::msgpack_codec codec;
    auto ret = codec.pack_args((int)rpc_service::result_code::OK, data.second);
    server_.response(conn_id, ret.data(), ret.size());
  }

  Channel<PredictParams>& InChannel() { return in_channel_; }

  bool IsEof() const { return is_eof_; }

 private:
  OFServing() : server_(9000, std::thread::hardware_concurrency()) { Init(); }

  int Add(rpc_service::connection* conn, int a, int b) { return a + b; }

  std::string Translate(rpc_service::connection* conn, const std::string& orignal) {
    std::string temp = orignal;
    for (auto& c : temp) c = toupper(c);
    return temp;
  }

  void Stop(rpc_service::connection* conn) {
    is_eof_ = false;
    in_channel_.Send({});
  }

  void PredictRaw(rpc_service::connection* conn, PredictParams params) {
    params.tag_id = conn->conn_id();
    in_channel_.Send(params);
  }

  void PredictJpeg(rpc_service::connection* conn, const PredictParams& params) {
    // todo
  }

  void Init() {
    server_.register_handler("add", &OFServing::Add, this);
    server_.register_handler("translate", &OFServing::Translate, this);
    server_.register_handler<ExecMode::async>("predict", &OFServing::PredictRaw, this);
    server_.register_handler<ExecMode::async>("predict_jpeg", &OFServing::PredictJpeg, this);
    server_.register_handler("stop", &OFServing::Stop, this);

    server_.run();
  }

  friend class Global<OFServing>;

  rpc_service::rpc_server server_;
  Channel<PredictParams> in_channel_;

  bool is_eof_ = true;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_OF_SERVING_H_
