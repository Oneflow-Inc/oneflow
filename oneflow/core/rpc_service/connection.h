#ifndef ONEFLOW_CORE_RPC_SERVICE_CONNECTION_H_
#define ONEFLOW_CORE_RPC_SERVICE_CONNECTION_H_

#include <iostream>
#include <memory>
#include <boost/asio.hpp>
#include <boost/asio/deadline_timer.hpp>
#include "oneflow/core/rpc_service/router.h"
using boost::asio::ip::tcp;

namespace oneflow {
namespace rpc_service {
class connection : public std::enable_shared_from_this<connection>, private boost::noncopyable {
 public:
  connection(boost::asio::io_service& io_service, std::size_t timeout_seconds)
      : socket_(io_service),
        data_(PAGE_SIZE),
        message_{boost::asio::buffer(head_), boost::asio::buffer(data_.data(), data_.size())},
        timer_(io_service),
        timeout_seconds_(timeout_seconds) {}

  ~connection() { close(); }

  void start() { read_head(); }

  tcp::socket& socket() { return socket_; }

  void response(const char* data, size_t len) {
    assert(message_[1].size() > len);
    message_[0] = boost::asio::buffer(&len, sizeof(int32_t));
    message_[1] = boost::asio::buffer((char*)data, len);
    reset_timer();
    auto self = this->shared_from_this();
    boost::asio::async_write(socket_, message_,
                             [this, self](boost::system::error_code ec, std::size_t length) {
                               cancel_timer();
                               if (!socket_.is_open()) { return; }
                               if (!ec) {
                                 read_head();
                               } else {
                                 LOG(INFO) << ec.message();
                               }
                             });
  }

 private:
  void read_head() {
    reset_timer();
    auto self(this->shared_from_this());
    boost::asio::async_read(socket_, boost::asio::buffer(head_),
                            [this, self](boost::system::error_code ec, std::size_t length) {
                              if (!socket_.is_open()) {
                                LOG(INFO) << "socket already closed";
                                return;
                              }

                              if (!ec) {
                                const int body_len = *((int*)(head_));
                                if (body_len > 0 && body_len < MAX_BUF_LEN) {
                                  if (data_.size() < body_len) { data_.resize(body_len); }
                                  read_body(body_len);
                                  return;
                                }

                                if (body_len == 0) {  // nobody, just head, maybe as heartbeat.
                                  cancel_timer();
                                  read_head();
                                } else {
                                  LOG(INFO) << "invalid body len";
                                  close();
                                }
                              } else {
                                LOG(INFO) << ec.message();
                                close();
                              }
                            });
  }

  void read_body(std::size_t size) {
    auto self(this->shared_from_this());
    boost::asio::async_read(socket_, boost::asio::buffer(data_.data(), size),
                            [this, self](boost::system::error_code ec, std::size_t length) {
                              cancel_timer();

                              if (!socket_.is_open()) {
                                LOG(INFO) << "socket already closed";
                                return;
                              }

                              if (!ec) {
                                router& _router = router::get();
                                _router.route(data_.data(), length, this);
                              } else {
                                LOG(INFO) << ec.message();
                              }
                            });
  }

  void reset_timer() {
    if (timeout_seconds_ == 0) { return; }

    auto self(this->shared_from_this());
    //    timer_.expires_at(timer_.expires_at() + boost::posix_time::seconds(5));
    timer_.expires_from_now(std::chrono::seconds(timeout_seconds_));
    timer_.async_wait([this, self](const boost::system::error_code& ec) {
      if (!socket_.is_open()) { return; }

      if (ec) { return; }

      LOG(INFO) << "rpc connection timeout";
      close();
    });
  }

  void cancel_timer() {
    if (timeout_seconds_ == 0) { return; }

    timer_.cancel();
  }

  void close() {
    if (socket_.is_open()) {
      boost::system::error_code ignored_ec;
      socket_.shutdown(tcp::socket::shutdown_both, ignored_ec);
      socket_.close(ignored_ec);
    }
  }

  tcp::socket socket_;
  char head_[HEAD_LEN];
  std::vector<char> data_;
  std::array<boost::asio::mutable_buffer, 2> message_;
  boost::asio::steady_timer timer_;
  std::size_t timeout_seconds_;
};
}  // namespace rpc_service
}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_SERVICE_CONNECTION_H_
