#include "oneflow/core/comm_network/epoll/socket_write_helper.h"
#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

#ifdef PLATFORM_POSIX

#include <sys/eventfd.h>

namespace oneflow {

SocketWriteHelper::~SocketWriteHelper() {
  delete cur_msg_queue_;
  cur_msg_queue_ = nullptr;
  {
    std::unique_lock<std::mutex> lck(pending_msg_queue_mtx_);
    delete pending_msg_queue_;
    pending_msg_queue_ = nullptr;
  }
}

SocketWriteHelper::SocketWriteHelper(int sockfd, IOEventPoller* poller) {
  sockfd_ = sockfd;
  queue_not_empty_fd_ = eventfd(0, 0);
  PCHECK(queue_not_empty_fd_ != -1);
  poller->AddFd(queue_not_empty_fd_,
                std::bind(&SocketWriteHelper::ProcessQueueNotEmptyEvent, this),
                [this]() {
                  // TODO: delete this log
                  LOG(INFO) << "fd " << queue_not_empty_fd_ << " writeable";
                });
  cur_msg_queue_ = new std::queue<SocketMsg>;
  pending_msg_queue_ = new std::queue<SocketMsg>;
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
  write_ptr_ = nullptr;
  write_size_ = 0;
}

void SocketWriteHelper::AsyncWrite(const SocketMsg& msg) {
  pending_msg_queue_mtx_.lock();
  bool need_send_event = pending_msg_queue_->empty();
  pending_msg_queue_->push(msg);
  pending_msg_queue_mtx_.unlock();
  if (need_send_event) { SendQueueNotEmptyEvent(); }
}

void SocketWriteHelper::NotifyMeSocketWriteable() { Work(); }

void SocketWriteHelper::SendQueueNotEmptyEvent() {
  uint64_t event_num = 1;
  PCHECK(write(queue_not_empty_fd_, &event_num, 8) == 8);
}

void SocketWriteHelper::ProcessQueueNotEmptyEvent() {
  uint64_t event_num = 0;
  PCHECK(read(queue_not_empty_fd_, &event_num, 8) == 8);
  Work();
}

void SocketWriteHelper::Work() {
  WriteUntilCurMsgQueueEmptyOrSocketNotWriteable();
  if (cur_msg_queue_->empty()) {
    {
      std::unique_lock<std::mutex> pending_lck(pending_msg_queue_mtx_);
      std::swap(cur_msg_queue_, pending_msg_queue_);
    }
    WriteUntilCurMsgQueueEmptyOrSocketNotWriteable();
  }
}

void SocketWriteHelper::WriteUntilCurMsgQueueEmptyOrSocketNotWriteable() {
  while ((this->*cur_write_handle_)()) {}
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
    void (SocketWriteHelper::*set_cur_write_done)()) {
  ssize_t n = write(sockfd_, write_ptr_, write_size_);
  if (n == write_size_) {
    (this->*set_cur_write_done)();
    return true;
  } else if (n >= 0) {
    write_ptr_ += n;
    write_size_ -= n;
    return true;
  } else {
    CHECK_EQ(n, -1);
    PCHECK(errno == EAGAIN || errno == EWOULDBLOCK);
    return false;
  }
}

void SocketWriteHelper::SetStatusWhenMsgHeadDone() {
  switch (cur_msg_.msg_type) {
#define MAKE_ENTRY(x, y) \
  case SocketMsgType::k##x: return SetStatusWhen##x##MsgHeadDone();
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ);
#undef MAKE_ENTRY
    default: UNEXPECTED_RUN();
  }
}

void SocketWriteHelper::SetStatusWhenMsgBodyDone() {
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
}

void SocketWriteHelper::SetStatusWhenRequestWriteMsgHeadDone() {
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
}

void SocketWriteHelper::SetStatusWhenRequestReadMsgHeadDone() {
  const void* src_token = cur_msg_.request_read_msg.src_token;
  auto src_mem_desc = static_cast<const SocketMemDesc*>(src_token);
  write_ptr_ = reinterpret_cast<const char*>(src_mem_desc->mem_ptr);
  write_size_ = src_mem_desc->byte_size;
  cur_write_handle_ = &SocketWriteHelper::MsgBodyWriteHandle;
}

void SocketWriteHelper::SetStatusWhenActorMsgHeadDone() {
  cur_write_handle_ = &SocketWriteHelper::InitMsgWriteHandle;
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
