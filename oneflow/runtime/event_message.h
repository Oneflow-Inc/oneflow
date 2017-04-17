#ifndef ONEFLOW_RUNTIME_EVENT_MESSAGE_H_
#define ONEFLOW_RUNTIME_EVENT_MESSAGE_H_
#include <glog/logging.h>
#include <cstdint>
#include <memory>

namespace oneflow {
enum class MessageType {
  kUnknown = 0,
  kConsumed,
  kProduced
};

class EventMessage {
 public:
  EventMessage() {}
  EventMessage(int32_t from_task_id, int32_t to_task_id, int64_t data_id,
    int64_t register_id, MessageType message_type) :
    from_task_id_(from_task_id), to_task_id_(to_task_id),
    data_id_(data_id), register_id_(register_id),
    message_type_(message_type), is_model_(false) {}
  ~EventMessage() {}

   EventMessage(const EventMessage& other) = default;
   EventMessage& operator=(const EventMessage& other) = default;

  // setter
  void set_from_task_id(int32_t from_task_id);
  void set_to_task_id(int32_t to_task_id);
  void set_data_id(int64_t data_id);
  void set_register_id(int64_t register_id);
  void set_message_type(MessageType message_type);
  void set_is_model(bool is_model);
  //void set_model_id(int64_t model_id);

  // getter
  int32_t from_task_id() const;
  int32_t to_task_id() const;
  int64_t data_id() const;
  int64_t register_id() const;
  MessageType message_type() const;
  bool is_model() const;
  //int64_t model_id() const;

 private:
  int32_t from_task_id_{ -1 };  // The task_id of sender
  int32_t to_task_id_{ -1 };    // The task_id of receiver
  int64_t data_id_{ -1 };       // On what data this message is generated
  int64_t register_id_{ -1 };   // The status change of which register
  MessageType message_type_;
  bool is_model_;
  //int64_t model_id_{ -1 };
};

typedef std::shared_ptr<EventMessage> MsgPtr;

inline void EventMessage::set_from_task_id(int32_t from_task_id) {
  from_task_id_ = from_task_id;
}

inline void EventMessage::set_to_task_id(int32_t to_task_id) {
  to_task_id_ = to_task_id;
}

inline void EventMessage::set_data_id(int64_t data_id) {
  data_id_ = data_id;
}

inline void EventMessage::set_register_id(int64_t register_id) {
  register_id_ = register_id;
}
inline void EventMessage::set_is_model(bool is_model) {
  is_model_ = is_model;
}
inline void EventMessage::set_message_type(MessageType message_type) {
  message_type_ = message_type;
}
//inline void EventMessage::set_model_id(int64_t model_id) {
//  model_id_ = model_id;
//}

inline MessageType EventMessage::message_type() const {
  CHECK(message_type_ != MessageType::kUnknown);
  return message_type_;
}

inline int32_t EventMessage::from_task_id() const {
  CHECK(from_task_id_ != -1);
  return from_task_id_;
}

inline int32_t EventMessage::to_task_id() const {
  CHECK(to_task_id_ != -1);
  return to_task_id_;
}

inline int64_t EventMessage::data_id() const {
  CHECK(data_id_ != -1);
  return data_id_;
}

inline int64_t EventMessage::register_id() const {
  CHECK(register_id_ != -1);
  return register_id_;
}
inline bool EventMessage::is_model() const {
  return is_model_;
}
//inline int64_t EventMessage::model_id() const {
//  CHECK(model_id_ != -1);
//  return model_id_;
//}
}  // namespace oneflow
#endif  // ONEFLOW_RUNTIME_EVENT_MESSAGE_H_
