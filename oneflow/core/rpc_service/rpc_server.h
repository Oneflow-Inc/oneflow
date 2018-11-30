#ifndef ONEFLOW_CORE_RPC_SERVICE_RPC_SERVER_H_
#define ONEFLOW_CORE_RPC_SERVICE_RPC_SERVER_H_

#include <thread>
#include "oneflow/core/rpc_service/connection.h"
#include "oneflow/core/rpc_service/io_service_pool.h"
#include "oneflow/core/rpc_service/router.h"

using boost::asio::ip::tcp;

namespace oneflow {
namespace rpc_service {
class rpc_server : private boost::noncopyable {
 public:
  rpc_server(short port, size_t size, size_t timeout_seconds = 0)
      : io_service_pool_(size),
        acceptor_(io_service_pool_.get_io_service(), tcp::endpoint(tcp::v4(), port)),
        timeout_seconds_(timeout_seconds) {
    router::get().set_callback(std::bind(&rpc_server::callback, this, std::placeholders::_1,
                                         std::placeholders::_2, std::placeholders::_3,
                                         std::placeholders::_4));
    do_accept();
  }

  ~rpc_server() {
    io_service_pool_.stop();
    thd_->join();
  }

  void run() {
    thd_ = std::make_shared<std::thread>([this] { io_service_pool_.run(); });
  }

  template<typename Function>
  void register_handler(std::string const& name, const Function& f) {
    router::get().register_handler(name, f);
  }

  template<ExecMode model = ExecMode::sync, typename Function, typename Self>
  void register_handler(std::string const& name, const Function& f, Self* self) {
    router::get().register_handler<model>(name, f, self);
  }

  void remove_handler(std::string const& name) { router::get().remove_handler(name); }

  void response(connection* conn, const char* data, size_t size) {
    std::unique_lock<std::mutex> lock(mtx_);
    auto it = connections_.begin();
    for (; it != connections_.end();) {
      std::shared_ptr<connection> conn_ptr = *it;
      if (!conn_ptr->socket().is_open()) {
        it = connections_.erase(it);
      } else {
        if (conn_ptr.get() == conn) {
          std::cout << "" << std::endl;
          conn_ptr->response(data, size);
        }
        ++it;
      }
    }
  }

 private:
  void do_accept() {
    conn_.reset(new connection(io_service_pool_.get_io_service(), timeout_seconds_));
    acceptor_.async_accept(conn_->socket(), [this](boost::system::error_code ec) {
      if (ec) {
        LOG(INFO) << "acceptor error: " << ec.message();
      } else {
        conn_->start();
        std::unique_lock<std::mutex> lock(mtx_);
        connections_.push_back(conn_);
      }

      do_accept();
    });
  }

 private:
  void callback(const std::string& topic, const std::string& result, connection* conn,
                bool has_error = false) {
    response(conn, result.data(), result.size());
  }

  io_service_pool io_service_pool_;
  tcp::acceptor acceptor_;
  std::shared_ptr<connection> conn_;
  std::shared_ptr<std::thread> thd_;
  std::size_t timeout_seconds_;

  std::vector<std::shared_ptr<connection>> connections_;
  std::mutex mtx_;
};
}  // namespace rpc_service
}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_SERVICE_RPC_SERVER_H_