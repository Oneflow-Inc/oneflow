package org.oneflow;

public class Option {

    // device
    private String deviceTag;
    private Boolean mirroredView;
    private Integer controlPort;

    // file
    private String savedModelDir;
    private String ModelVersion;
    private String metaFileBaseName;
    private String fullPathName;
    private String checkpointDir;

    // config
    private String graphName;
    private String signatureName;
    private Integer batchSize;

    public Option() {
    }

    public String getDeviceTag() {
        return deviceTag;
    }

    public Option setDeviceTag(String deviceTag) {
        this.deviceTag = deviceTag;
        return this;
    }

    public Boolean getMirroredView() {
        return mirroredView;
    }

    public Option setMirroredView(Boolean mirroredView) {
        this.mirroredView = mirroredView;
        return this;
    }

    public Integer getControlPort() {
        return controlPort;
    }

    public Option setControlPort(Integer controlPort) {
        this.controlPort = controlPort;
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
        return ModelVersion;
    }

    public Option setModelVersion(String modelVersion) {
        ModelVersion = modelVersion;
        return this;
    }

    public String getMetaFileBaseName() {
        return metaFileBaseName;
    }

    public Option setMetaFileBaseName(String metaFileBaseName) {
        this.metaFileBaseName = metaFileBaseName;
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

    public Integer getBatchSize() {
        return batchSize;
    }

    public Option setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public String getCheckpointDir() {
        return checkpointDir;
    }

    public String getFullPathName() {
        return fullPathName;
    }

    public Option setFullPathName(String fullPathName) {
        this.fullPathName = fullPathName;
        return this;
    }

    @Override
    public String toString() {
        return "Option{" +
                "deviceTag='" + deviceTag + '\'' +
                ", mirroredView=" + mirroredView +
                ", controlPort=" + controlPort +
                ", savedModelDir='" + savedModelDir + '\'' +
                ", ModelVersion=" + ModelVersion +
                ", metaFileBaseName='" + metaFileBaseName + '\'' +
                ", graphName='" + graphName + '\'' +
                ", signatureName='" + signatureName + '\'' +
                ", batchSize=" + batchSize +
                '}';
    }
}
