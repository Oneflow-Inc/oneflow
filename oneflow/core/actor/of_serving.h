#ifndef ONEFLOW_CORE_ACTOR_OF_SERVING_H_
#define ONEFLOW_CORE_ACTOR_OF_SERVING_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/actor/sink_compute_actor.h"
#include "oneflow/core/rpc_service/rpc_server.h"
#include "oneflow/core/rpc_service/connection.h"

namespace oneflow {

using ConnBufferPair = std::pair<int64_t, std::vector<rpc_service::blob_t>>;

class OFServing final : public SinkCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFServing);

  void SendMsg(const std::pair<int64_t, std::string>& data) {
    //    if (callback_ == nullptr) {
    //      std::cout << "finish" << std::endl;
    //      return;
    //    }

    int64_t conn_id = data.first;
    rpc_service::connection* conn = reinterpret_cast<rpc_service::connection*>(conn_id);
    rpc_service::msgpack_codec codec;
    auto ret = codec.pack_args((int)rpc_service::result_code::OK, data.second);
    server_.response(conn, ret.data(), ret.size());
  }

  Channel<ConnBufferPair>& InChannel() { return in_channel_; }

  bool IsEof() const { return is_eof_; }

 private:
  OFServing() : server_(9000, std::thread::hardware_concurrency()) { Init(); }

  int add(rpc_service::connection* conn, int a, int b) { return a + b; }

  std::string translate(rpc_service::connection* conn, const std::string& orignal) {
    std::string temp = orignal;
    for (auto& c : temp) c = toupper(c);
    return temp;
  }

  void stop(rpc_service::connection* conn) {
    is_eof_ = false;
    in_channel_.Send({});
  }

  void Predict(rpc_service::connection* conn, const std::string& version,
               const std::vector<rpc_service::blob_t>& blobs) {
    in_channel_.Send(std::make_pair(reinterpret_cast<int64_t>(conn), blobs));
  }

  void Init() {
    server_.register_handler("add", &OFServing::add, this);
    server_.register_handler("translate", &OFServing::translate, this);
    server_.register_handler<ExecMode::async>("predict", &OFServing::Predict, this);
    server_.register_handler("stop", &OFServing::stop, this);

    server_.run();
  }

  friend class Global<OFServing>;

  //  std::function<void(const std::vector<std::pair<const char*, size_t>>&)> callback_;

  rpc_service::rpc_server server_;
  Channel<ConnBufferPair> in_channel_;

  bool is_eof_ = true;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_OF_SERVING_H_
