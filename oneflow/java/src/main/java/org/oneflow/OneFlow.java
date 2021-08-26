package org.oneflow;

import java.nio.Buffer;
import java.nio.ByteOrder;

class OneFlow {
    static {
        System.loadLibrary("oneflow_java");

        // Default Initialization: beyond import oneflow as flow
        OneFlow.initDefaultSession();
        if (getEndian() == 0) {
            Tensor.endian = ByteOrder.BIG_ENDIAN;
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("shutting down OneFlow");
            OneFlow.destroyEnv();
            OneFlow.setShuttingDown();
        }));
    }

    // 0 for big endian, 1 for little endian
    static native int getEndian();

    // init
    static native void setIsMultiClient(boolean isMultiClient);
    static native void initDefaultSession();
    static native boolean isEnvInited();
    static native void initEnv(int port);
    static native long currentMachineId();
    static native void initScopeStack();
    static native boolean isSessionInited();
    static native void initSession(String deviceTag);

    // compile
    static native void loadModel(Option option);

    // launch
    static native void startLazyGlobalSession();
    static native void loadCheckpoint(Buffer path);

    // forward
    static native String getPushJobNames();
    static native String getPullJobNames();
    static native void runSinglePushJob(Buffer data,
                                        Buffer shape,
                                        int dTypeCode,
                                        String jobName,
                                        String opName);
    static native void runInferenceJob(String jobName);
    static native Tensor runPullJob(String jobName, String opName);

    // clean
    static native void stopLazyGlobalSession();
    static native void destroyLazyGlobalSession();
    static native void destroyEnv();
    static native void setShuttingDown();
}
