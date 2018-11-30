#ifndef ONEFLOW_CORE_RPC_SERVICE_ROUTER_H_
#define ONEFLOW_CORE_RPC_SERVICE_ROUTER_H_

#include <boost/asio.hpp>
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/rpc_service/common.h"
#include "oneflow/core/rpc_service/codec.h"
#include "oneflow/core/rpc_service/codec.h"

namespace oneflow {
enum class ExecMode { sync, async };

namespace rpc_service {
class connection;

class router : boost::noncopyable {
 public:
  static router& get() {
    static router instance;
    return instance;
  }

  template<typename Function>
  void register_handler(std::string const& name, const Function& f) {
    return register_nonmember_func(name, f);
  }

  template<ExecMode model, typename Function, typename Self>
  void register_handler(std::string const& name, const Function& f, Self* self) {
    return register_member_func<model>(name, f, self);
  }

  void remove_handler(std::string const& name) { this->map_invokers_.erase(name); }

  void set_callback(const std::function<void(const std::string&, const std::string&, connection*,
                                             bool)>& callback) {
    callback_to_server_ = callback;
  }

  template<typename T>
  void route(const char* data, std::size_t size, T conn) {
    std::string result;
    msgpack_codec codec;
    auto p = codec.unpack<std::tuple<std::string>>(data, size);
    auto func_name = std::get<0>(p);
    auto it = map_invokers_.find(func_name);
    if (it == map_invokers_.end()) {
      result = codec.pack_args_str(result_code::FAIL, "unknown function: " + func_name);
      callback_to_server_(func_name, result, conn, true);
      return;
    }

    ExecMode model;
    it->second(conn, data, size, result, model);
    if (model == ExecMode::sync && callback_to_server_) {
      callback_to_server_(func_name, result, conn, false);
    }
  }

  router() = default;

 private:
  router(const router&) = delete;
  router(router&&) = delete;

  template<typename F, size_t... I, typename Arg, typename... Args>
  static typename std::result_of<F(Args...)>::type call_helper(
      const F& f, const std::index_sequence<I...>&, const std::tuple<Arg, Args...>& tup) {
    return f(std::get<I + 1>(tup)...);
  }

  template<typename F, typename Arg, typename... Args>
  static
      typename std::enable_if<std::is_void<typename std::result_of<F(Args...)>::type>::value>::type
      call(const F& f, std::string& result, std::tuple<Arg, Args...>& tp) {
    call_helper(f, std::make_index_sequence<sizeof...(Args)>{}, tp);
    result = msgpack_codec::pack_args_str(result_code::OK);
  }

  template<typename F, typename Arg, typename... Args>
  static
      typename std::enable_if<!std::is_void<typename std::result_of<F(Args...)>::type>::value>::type
      call(const F& f, std::string& result, const std::tuple<Arg, Args...>& tp) {
    auto r = call_helper(f, std::make_index_sequence<sizeof...(Args)>{}, tp);
    msgpack_codec codec;
    result = msgpack_codec::pack_args_str(result_code::OK, r);
  }

  template<typename F, typename Self, size_t... Indexes, typename Arg, typename... Args>
  static typename std::result_of<F(Self, connection*, Args...)>::type call_member_helper(
      const F& f, Self* self, const std::index_sequence<Indexes...>&,
      const std::tuple<Arg, Args...>& tup, connection* ptr = 0) {
    return (*self.*f)(ptr, std::get<Indexes + 1>(tup)...);
  }

  template<typename F, typename Self, typename Arg, typename... Args>
  static typename std::enable_if<
      std::is_void<typename std::result_of<F(Self, connection*, Args...)>::type>::value>::type
  call_member(const F& f, Self* self, connection* ptr, std::string& result,
              const std::tuple<Arg, Args...>& tp) {
    call_member_helper(f, self, typename std::make_index_sequence<sizeof...(Args)>{}, tp, ptr);
    result = msgpack_codec::pack_args_str(result_code::OK);
  }

  template<typename F, typename Self, typename Arg, typename... Args>
  static typename std::enable_if<
      !std::is_void<typename std::result_of<F(Self, connection*, Args...)>::type>::value>::type
  call_member(const F& f, Self* self, connection* ptr, std::string& result,
              const std::tuple<Arg, Args...>& tp) {
    auto r = call_member_helper(f, self, typename std::make_index_sequence<sizeof...(Args)>{}, tp);
    result = msgpack_codec::pack_args_str(result_code::OK, r);
  }

  template<typename Function, ExecMode mode = ExecMode::sync>
  struct invoker {
    static inline void apply(const Function& func, connection* conn, const char* data, size_t size,
                             std::string& result) {
      using args_tuple = typename function_traits<Function>::args_tuple;

      msgpack_codec codec;
      try {
        auto tp = codec.unpack<args_tuple>(data, size);
        call(func, result, tp);
      } catch (std::invalid_argument& e) {
        result = codec.pack_args_str(result_code::FAIL, e.what());
      } catch (const std::exception& e) {
        result = codec.pack_args_str(result_code::FAIL, e.what());
      }
    }

    template<ExecMode model, typename Self>
    static inline void apply_member(const Function& func, Self* self, connection* conn,
                                    const char* data, size_t size, std::string& result,
                                    ExecMode& exe_model) {
      using args_tuple = typename function_traits<Function>::args_tuple_2nd;
      exe_model = ExecMode::sync;
      msgpack_codec codec;
      try {
        auto tp = codec.unpack<args_tuple>(data, size);
        call_member(func, self, conn, result, tp);
        exe_model = model;
      } catch (std::invalid_argument& e) {
        result = codec.pack_args_str(result_code::FAIL, e.what());
      } catch (const std::exception& e) {
        result = codec.pack_args_str(result_code::FAIL, e.what());
      }
    }
  };

  template<typename Function>
  void register_nonmember_func(std::string const& name, const Function& f) {
    this->map_invokers_[name] = {std::bind(&invoker<Function>::apply, f, std::placeholders::_1,
                                           std::placeholders::_2, std::placeholders::_3,
                                           std::placeholders::_4)};
  }

  template<ExecMode model, typename Function, typename Self>
  void register_member_func(const std::string& name, const Function& f, Self* self) {
    this->map_invokers_[name] = {std::bind(&invoker<Function>::template apply_member<model, Self>,
                                           f, self, std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3, std::placeholders::_4,
                                           std::placeholders::_5)};
  }

  std::map<std::string,
           std::function<void(connection*, const char*, size_t, std::string&, ExecMode& model)>>
      map_invokers_;
  std::function<void(const std::string&, const std::string&, connection*, bool)>
      callback_to_server_;
};
}  // namespace rpc_service
}  // namespace oneflow

#endif  // ONEFLOW_CORE_RPC_SERVICE_ROUTER_H_
