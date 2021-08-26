package org.oneflow;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;


public class InferenceSession {

    private final Option option;

    private final static String CLOSE = "CLOSE";
    private final static String OPEN = "OPEN";
    private String status = CLOSE;

    public InferenceSession(Option option) {
        this.option = option;
    }

    public void open() throws RuntimeException {
        checkStatus(CLOSE);
        checkOption();

        OneFlow.setIsMultiClient(false);

        if (!OneFlow.isEnvInited()) {
            OneFlow.initEnv(option.getControlPort());
            if (OneFlow.currentMachineId() == 0) {
                OneFlow.initScopeStack();
            }
        }

        if (!OneFlow.isSessionInited()) {
            OneFlow.initSession(option.getDeviceTag());
        }

        status = OPEN;
        loadModel(option.getSavedModelDir());
        launch();
    }

    private void loadModel(String path) throws RuntimeException {
        String version = option.getModelVersion();
        String savedModelPath = path + File.separator + version + File.separator;
        option.setModelProtoPath(savedModelPath + File.separator + option.getMetaFileBaseName());
        checkModelProtoFile();
        OneFlow.loadModel(option);
    }

    private void launch() throws RuntimeException {
        OneFlow.startLazyGlobalSession();

        String path = option.getSavedModelDir() + File.separator +
                option.getModelVersion() + File.separator +
                option.getCheckpointDir();
        checkCheckpointDir(path);
        byte[] checkpointBytes = path.getBytes();
        ByteBuffer checkpointBuffer = ByteBuffer.allocateDirect(checkpointBytes.length);
        checkpointBuffer.put(checkpointBytes);

        OneFlow.loadCheckpoint(checkpointBuffer);
    }

    public synchronized Map<String, Tensor> run(String jobName, Map<String, Tensor> tensorMap) throws RuntimeException {
        checkStatus(OPEN);
        // push job names: [job, op, job, op, ...]
        String[] pushJobNames = OneFlow.getPushJobNames().split(",");
        for (int i = 0; i < pushJobNames.length; i += 2) {
            Tensor tensor = tensorMap.get(pushJobNames[i]);
            OneFlow.runSinglePushJob(tensor.getDataBuffer(), tensor.getShapeBuffer(),
                    tensor.getDataType().code, pushJobNames[i + 1], pushJobNames[i]);
        }

        // Inference
        OneFlow.runInferenceJob(jobName);

        // Pull
        String[] pullJobNames = OneFlow.getPullJobNames().split(",");
        Map<String, Tensor> resultMap = new HashMap<>();
        for (int i = 0; i < pullJobNames.length; i += 2) {
            Tensor res = OneFlow.runPullJob(pullJobNames[i + 1], pullJobNames[i]);
            resultMap.put(pullJobNames[i], res);
        }

        return resultMap;
    }

    public void close() throws RuntimeException {
        checkStatus(OPEN);

        OneFlow.stopLazyGlobalSession();
        OneFlow.destroyLazyGlobalSession();

        status = CLOSE;
    }

    private void checkOption() throws RuntimeException {
        if (option.getDeviceTag() == null) {
            throw new RuntimeException("device tag cannot be null");
        }
        else if (option.getSavedModelDir() == null) {
            throw new RuntimeException("saved model dir cannot be null");
        }
        else if (option.getControlPort() == null) {
            throw new RuntimeException("control port cannot be null");
        }
        else if (option.getModelVersion() == null) {
            throw new RuntimeException("model version cannot be null");
        }
    }

    private void checkModelProtoFile() throws RuntimeException {
        File protoFile = new File(option.getModelProtoPath());
        if (!protoFile.exists()) {
            throw new RuntimeException(".pb file is not exist");
        }
    }

    private void checkCheckpointDir(String path) throws RuntimeException {
        File checkpointDir = new File(path);
        if (!checkpointDir.exists()) {
            throw new RuntimeException("checkpoint dir is not exist");
        }
    }

    private void checkStatus(String expected) throws RuntimeException {
        if (!status.equals(expected)) {
            throw new RuntimeException("session status check failed");
        }
    }
}
