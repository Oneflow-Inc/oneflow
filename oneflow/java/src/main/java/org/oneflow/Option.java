package org.oneflow;

import java.util.Objects;

public class Option {

    private final static String SAVED_MODEL_PB = "saved_model.pb";
    private final static String MACHINE_DEVICE_IDS = "0:0";
    private final static Boolean MIRRORED_VIEW = false;

    // Must
    private String deviceTag;
    private String savedModelDir;
    private String modelVersion;
    private Integer controlPort;

    // Option, can be null
    private Integer batchSize;

    // Option, will be determined at runtime if not given
    private String graphName;
    private String signatureName;

    // Option, default value will be provided if users do not given one
    private String machineDeviceIds;
    private String metaFileBaseName;
    private Boolean mirroredView;

    // used by JNI, you need to change the native code when change these names.
    private String modelProtoPath;
    private String checkpointDir;

    public Option() {
        this.metaFileBaseName = SAVED_MODEL_PB;
        this.machineDeviceIds = MACHINE_DEVICE_IDS;
        this.mirroredView = MIRRORED_VIEW;
    }

    public String getDeviceTag() {
        return deviceTag;
    }

    public Option setDeviceTag(String deviceTag) {
        this.deviceTag = deviceTag;
        return this;
    }

    public String getSavedModelDir() {
        return savedModelDir;
    }

    public Option setSavedModelDir(String savedModelDir) {
        this.savedModelDir = savedModelDir;
        return this;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public Option setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
        return this;
    }

    public Integer getControlPort() {
        return controlPort;
    }

    public Option setControlPort(Integer controlPort) {
        this.controlPort = controlPort;
        return this;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public Option setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public String getGraphName() {
        return graphName;
    }

    public Option setGraphName(String graphName) {
        this.graphName = graphName;
        return this;
    }

    public String getSignatureName() {
        return signatureName;
    }

    public Option setSignatureName(String signatureName) {
        this.signatureName = signatureName;
        return this;
    }

    public String getMachineDeviceIds() {
        return machineDeviceIds;
    }

    public Option setMachineDeviceIds(String machineDeviceIds) {
        this.machineDeviceIds = machineDeviceIds;
        return this;
    }

    public String getMetaFileBaseName() {
        return metaFileBaseName;
    }

    public Option setMetaFileBaseName(String metaFileBaseName) {
        this.metaFileBaseName = metaFileBaseName;
        return this;
    }

    public Boolean getMirroredView() {
        return mirroredView;
    }

    public Option setMirroredView(Boolean mirroredView) {
        this.mirroredView = mirroredView;
        return this;
    }

    public String getModelProtoPath() {
        return modelProtoPath;
    }

    public Option setModelProtoPath(String modelProtoPath) {
        this.modelProtoPath = modelProtoPath;
        return this;
    }

    public String getCheckpointDir() {
        return checkpointDir;
    }

    public Option setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
        return this;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Option option = (Option) o;
        return Objects.equals(deviceTag, option.deviceTag) &&
                Objects.equals(savedModelDir, option.savedModelDir) &&
                Objects.equals(modelVersion, option.modelVersion) &&
                Objects.equals(controlPort, option.controlPort) &&
                Objects.equals(batchSize, option.batchSize) &&
                Objects.equals(graphName, option.graphName) &&
                Objects.equals(signatureName, option.signatureName) &&
                Objects.equals(machineDeviceIds, option.machineDeviceIds) &&
                Objects.equals(metaFileBaseName, option.metaFileBaseName) &&
                Objects.equals(mirroredView, option.mirroredView) &&
                Objects.equals(modelProtoPath, option.modelProtoPath) &&
                Objects.equals(checkpointDir, option.checkpointDir);
    }

    @Override
    public int hashCode() {
        return Objects.hash(deviceTag, savedModelDir, modelVersion, controlPort, batchSize, graphName, signatureName, machineDeviceIds, metaFileBaseName, mirroredView, modelProtoPath, checkpointDir);
    }

    @Override
    public String toString() {
        return "Option{" +
                "deviceTag='" + deviceTag + '\'' +
                ", savedModelDir='" + savedModelDir + '\'' +
                ", modelVersion='" + modelVersion + '\'' +
                ", controlPort=" + controlPort +
                ", batchSize=" + batchSize +
                ", graphName='" + graphName + '\'' +
                ", signatureName='" + signatureName + '\'' +
                ", machineDeviceIds='" + machineDeviceIds + '\'' +
                ", metaFileBaseName='" + metaFileBaseName + '\'' +
                ", mirroredView=" + mirroredView +
                ", modelProtoPath='" + modelProtoPath + '\'' +
                ", checkpointDir='" + checkpointDir + '\'' +
                '}';
    }
}
