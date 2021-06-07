#include "oneflow/api/java/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/env/env.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/api/java/util/string_util.h"

JNIEXPORT
jboolean JNICALL Java_org_oneflow_env_Env_isEnvInited(JNIEnv* env, jobject obj) {
  return oneflow::IsEnvInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_env_Env_initEnv(JNIEnv* env, jobject obj, jstring env_proto_str) {
  return oneflow::InitEnv(convert_jstring_to_string(env, env_proto_str)).GetOrThrow();
}
