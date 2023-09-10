import logging

import pytest
from tr_learn.log_set import init_logging

init_logging("./log_settings.yaml")


def test_logging(caplog):
    root_logger = logging.getLogger()
    text = "Test"
    root_logger.info(text)
    assert text in caplog.records[-1].message
    assert caplog.records[-1].levelno == logging.INFO
