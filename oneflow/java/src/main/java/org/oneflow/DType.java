package org.oneflow;

/**
 * Reference: oneflow/core/common/data_type.proto
 */
public enum DType {
    kInvalidDataType(0, -1),
    kChar(1, 1),
    kFloat(2, 4),
    kDouble(3, 8),
    kInt8(4, 1),
    kInt32(5, 4),
    kInt64(6, 8),
    kUInt8(7, 1),
    kOFRecord(8, -1),
    kFloat16(9, 2),
    kTensorBuffer(10, -1);

    public final int code;
    public final int bytes;

    DType(int code, int bytes) {
        this.code = code;
        this.bytes = bytes;
    }
}
