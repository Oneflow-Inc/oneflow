package org.oneflow;


import org.oneflow.tensor.FloatTensor;
import org.oneflow.tensor.IntTensor;

import java.nio.*;


public abstract class Tensor {
    static ByteOrder endian = ByteOrder.LITTLE_ENDIAN;

    private final long[] shape;

    protected Tensor(long[] shape) {
        this.shape = shape;
    }

    public static Tensor fromBlob(int[] data, long[] shape) {
        final IntBuffer intBuffer = ByteBuffer.allocateDirect(data.length * DType.kInt32.bytes)
                .order(Tensor.endian)
                .asIntBuffer();
        intBuffer.put(data);
        return new IntTensor(shape, intBuffer);
    }

    public static Tensor fromBlob(float[] data, long[] shape) {
        final FloatBuffer floatBuffer = ByteBuffer.allocateDirect(data.length * DType.kFloat.bytes)
                .order(Tensor.endian)
                .asFloatBuffer();
        floatBuffer.put(data);
        return new FloatTensor(shape, floatBuffer);
    }

    public long[] getShape() {
        return shape;
    }

    /**
     * The byte order will depend on host machine, for x86, it will be little endian
     * @return specifically It is a DirectBuffer
     */
    Buffer getShapeBuffer() {
        LongBuffer buffer = ByteBuffer.allocateDirect(shape.length * Long.BYTES)
                .order(endian)
                .asLongBuffer();
        buffer.put(shape);
        return buffer;
    }

    public abstract DType getDataType();

    public abstract Buffer getDataBuffer();

    public boolean[] getDataAsBooleanArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as boolean array");
    }

    public byte[] getDataAsByteArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as byte array");
    }

    public short[] getDataAsShortArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as short array");
    }

    public int[] getDataAsIntArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as int array");
    }

    public long[] getDataAsLongArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as long array");
    }

    public float[] getDataAsFloatArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as float array");
    }

    public double[] getDataAsDoubleArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as double array");
    }

    /**
     * This function will be called from native code, so when the function
     * signature changed, you need to changed the native code too
     * command: javap -s Tensor.class
     */
    static Tensor nativeNewTensor(byte[] data, long[] shape, int dType) {
        // Todo: why not call fromBlob?
        Tensor tensor = null;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(data.length);
        byteBuffer.put(data);
        byteBuffer.rewind();

        if (DType.kFloat.code == dType) {
            tensor = new FloatTensor(shape, byteBuffer.order(endian).asFloatBuffer());
        }
        else if (DType.kInt32.code == dType) {
            tensor = new IntTensor(shape, byteBuffer.order(endian).asIntBuffer());
        }

        return tensor;
    }
}
