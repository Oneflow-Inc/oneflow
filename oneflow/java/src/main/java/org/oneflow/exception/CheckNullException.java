package org.oneflow.exception;

public class CheckNullException extends RuntimeException {

    public CheckNullException() {
        super();
    }

    public CheckNullException(String message, Throwable cause) {
        super(message, cause);
    }

    public CheckNullException(String message) {
        super(message);
    }

    public CheckNullException(Throwable cause) {
        super(cause);
    }

}
