#include "oneflow/core/comm_network/epoll/socket_write_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketWriteHelper::~SocketWriteHelper() {
  {
    std::unique_lock<std::mutex> lck(cur_msg_queue_mtx_);
    delete cur_msg_queue_;
    cur_msg_queue_ = nullptr;
  }
  {
    std::unique_lock<std::mutex> lck(pending_msg_queue_mtx_);
    delete pending_msg_queue_;
    pending_msg_queue_ = nullptr;
  }
}

SocketWriteHelper::SocketWriteHelper(int sockfd, CpuStream* cpu_stream) {
  sockfd_ = sockfd;
  cpu_stream_ = cpu_stream;
  cur_msg_queue_ = new std::queue<SocketMsg>;
  pending_msg_queue_ = new std::queue<SocketMsg>;
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
  write_ptr_ = nullptr;
  write_size_ = 0;
}

void SocketWriteHelper::AsyncWrite(const SocketMsg& msg) {
  if (cur_msg_queue_mtx_.try_lock()) {
    bool need_notify_worker = cur_msg_queue_->empty();
    cur_msg_queue_->push(msg);
    cur_msg_queue_mtx_.unlock();
    if (need_notify_worker) { NotifyWorker(); }
  } else {
    std::unique_lock<std::mutex> lck(pending_msg_queue_mtx_);
    pending_msg_queue_->push(msg);
  }
}

void SocketWriteHelper::NotifyMeSocketWriteable() { NotifyWorker(); }

void SocketWriteHelper::WriteUntilCurMsgQueueEmptyOrSocketNotWriteable() {
  while ((this->*cur_write_handle_)()) {}
}

void SocketWriteHelper::NotifyWorker() {
  cpu_stream_->SendWork([this]() {
    std::unique_lock<std::mutex> cur_lck(cur_msg_queue_mtx_);
    WriteUntilCurMsgQueueEmptyOrSocketNotWriteable();
    if (cur_msg_queue_->empty()) {
      {
        std::unique_lock<std::mutex> pending_lck(pending_msg_queue_mtx_);
        std::swap(cur_msg_queue_, pending_msg_queue_);
      }
      WriteUntilCurMsgQueueEmptyOrSocketNotWriteable();
    }
  });
}

bool SocketWriteHelper::InitMsgWriteHandle() {
  if (cur_msg_queue_->empty()) { return false; }
  cur_msg_ = cur_msg_queue_->front();
  cur_msg_queue_->pop();
  write_ptr_ = reinterpret_cast<const char*>(&cur_msg_);
  write_size_ = sizeof(cur_msg_);
  cur_write_handle_ = &SocketWriteHelper::MsgHeadWriteHandle;
  return true;
}

bool SocketWriteHelper::MsgHeadWriteHandle() {
  return DoCurWrite(&SocketWriteHelper::SetStatusWhenMsgHeadDone);
}

bool SocketWriteHelper::MsgBodyWriteHandle() {
  return DoCurWrite(&SocketWriteHelper::SetStatusWhenMsgBodyDone);
}

bool SocketWriteHelper::DoCurWrite(
    bool (SocketWriteHelper::*set_cur_write_done)()) {
  ssize_t n = write(sockfd_, write_ptr_, write_size_);
  if (n == write_size_) {
    return (this->*set_cur_write_done)();
  } else if (n >= 0) {
    write_ptr_ = write_ptr_ + n;
    write_size_ -= n;
    return true;
  } else {
    CHECK_EQ(n, -1);
    PCHECK(errno == EAGAIN || errno == EWOULDBLOCK);
    return false;
  }
}

bool SocketWriteHelper::SetStatusWhenMsgHeadDone() { TODO(); }

bool SocketWriteHelper::SetStatusWhenMsgBodyDone() {
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
  return true;
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
