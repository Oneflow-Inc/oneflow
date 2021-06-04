#include <stdio.h>
#include "example.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/env/env.h"

JNIEXPORT jboolean JNICALL Java_com_example_App_isEnvInited(JNIEnv *, jobject) {
    oneflow::Maybe<bool> x = oneflow::IsEnvInited();
    printf("call IsEnvInited()\n");
    if (x.IsOk()) {
        return x.GetOrThrow();
    }
    else {
        return false;
    }
}

JNIEXPORT void JNICALL Java_com_example_App_initEnv(
    JNIEnv *env, jobject obj, jstring envProto) {
    const char *str;
    str = env->GetStringUTFChars(envProto, NULL);
    std::string envProtoStr = str;
    std::cout << str << std::endl;
    return oneflow::InitEnv(envProtoStr).GetOrThrow();
}

JNIEXPORT void JNICALL Java_com_example_App_helloWorld(JNIEnv *, jobject) {
    printf("hello world\n");
}

int fibonacii(int x) {
    if (x == 0 || x == 1) {
        return 1;
    }
    return fibonacii(x - 1) + fibonacii(x - 2);
}

JNIEXPORT jint JNICALL Java_com_example_App_fibonacii(JNIEnv * env, jobject obj, jint num) {
    return fibonacii(num);
}