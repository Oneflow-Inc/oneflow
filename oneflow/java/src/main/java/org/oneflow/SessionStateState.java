package org.oneflow;

public enum SessionStateState {
    OPEN(0, "open"),
    RUNNING(1, "running"),
    CLOSE(2, "closed");

    final int code;
    final String desc;

    SessionStateState(int code, String desc) {
        this.code = code;
        this.desc = desc;
    }
}
