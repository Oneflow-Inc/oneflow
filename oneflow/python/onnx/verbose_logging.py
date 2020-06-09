# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
A wrapper of built-in logging with custom level support and utilities.
"""

from contextlib import contextmanager
import logging as _logging
from logging import *  # pylint: disable=wildcard-import, unused-wildcard-import
import os
import types

from . import constants

VERBOSE = 15

_logging.addLevelName(VERBOSE, "VERBOSE")


def _verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kwargs)  # pylint: disable=protected-access


def getLogger(name=None):  # pylint: disable=invalid-name, function-redefined
    logger = _logging.getLogger(name)
    # Inject verbose method to logger object instead logging module
    logger.verbose = types.MethodType(_verbose, logger)
    return logger


_BASIC_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_VERBOSE_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"


def basicConfig(**kwargs):  # pylint: disable=invalid-name, function-redefined
    """ Do basic configuration for the logging system. tf verbosity is updated accordingly. """
    # Choose pre-defined format if format argument is not specified
    if "format" not in kwargs:
        level = kwargs.get("level", _logging.root.level)
        kwargs["format"] = _BASIC_LOG_FORMAT if level >= INFO else _VERBOSE_LOG_FORMAT
    # config will make effect only when root.handlers is empty, so add the following statement to make sure it
    _logging.root.handlers = []
    _logging.basicConfig(**kwargs)


_LOG_LEVELS = [FATAL, ERROR, WARNING, INFO, VERBOSE, DEBUG]


def get_verbosity_level(verbosity, base_level=INFO):
    """ If verbosity is specified, return corresponding level, otherwise, return default_level. """
    if verbosity is None:
        return base_level
    verbosity = min(max(0, verbosity) + _LOG_LEVELS.index(base_level), len(_LOG_LEVELS) - 1)
    return _LOG_LEVELS[verbosity]


def set_level(level):
    """ Set logging level for oneflow.python.onnx package. tf verbosity is updated accordingly. """
    _logging.getLogger(constants.TF2ONNX_PACKAGE_NAME).setLevel(level)


@contextmanager
def set_scope_level(level, logger=None):
    """
    Set logging level to logger within context, reset level to previous value when exit context.
    TF verbosity is NOT affected.
    """
    if logger is None:
        logger = getLogger()

    current_level = logger.level
    logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(current_level)
