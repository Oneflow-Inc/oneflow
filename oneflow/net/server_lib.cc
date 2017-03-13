#include "server_lib.h"

#include <unordered_map>


namespace oneflow {

namespace {

typedef std::unordered_map<std::string, ServerFactory*> ServerFactories;
ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}
}  // namespace

/* static */
void ServerFactory::Register(const std::string& server_type,
                             ServerFactory* factory) {
  if (!server_factories()->insert({server_type, factory}).second) {
    std::cout << "Two server factories are being registered under "
               << server_type;
  }
}

/* static */
int ServerFactory::GetFactory(const ServerDef& server_def,
                                 ServerFactory** out_factory) {
  // TODO(mrry): Improve the error reporting here.
  for (const auto& server_factory : *server_factories()) {
    if (server_factory.second->AcceptsOptions(server_def)) {
      *out_factory = server_factory.second;
      return 1;
    }
  }
  return 0;
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
int NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  //TF_RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  ServerFactory::GetFactory(server_def, &factory);
  return factory->NewServer(server_def, out_server);
}

}  // namespace oneflow
