package org.oneflow.exception;

public class FileNotExistException extends RuntimeException {

    public FileNotExistException() {
        super();
    }

    public FileNotExistException(String message, Throwable cause) {
        super(message, cause);
    }

    public FileNotExistException(String message) {
        super(message);
    }

    public FileNotExistException(Throwable cause) {
        super(cause);
    }

}
