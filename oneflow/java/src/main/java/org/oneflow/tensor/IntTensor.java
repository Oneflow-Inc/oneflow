package org.oneflow.tensor;

import org.oneflow.DType;
import org.oneflow.Tensor;

import java.nio.Buffer;
import java.nio.IntBuffer;

public class IntTensor extends Tensor {
    private final IntBuffer data;

    public IntTensor(long[] shape, IntBuffer data) {
        super(shape);
        this.data = data;
    }

    @Override
    public DType getDataType() {
        return DType.kInt32;
    }

    @Override
    public Buffer getDataBuffer() {
        data.rewind();
        return data;
    }

    @Override
    public int[] getDataAsIntArray() {
        data.rewind();
        int[] arr = new int[data.remaining()];
        data.get(arr);
        return arr;
    }

}
