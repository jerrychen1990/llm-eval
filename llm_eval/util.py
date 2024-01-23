#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/23 13:53:55
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''
import logging
import os
from snippets.logs import getlog_detail

LLM_EVAL_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
LOG_DIR = os.path.join(LLM_EVAL_DIR, "logs")
print(LOG_DIR)

def get_logger(name):
    logger = getlog_detail(name=name, level=logging.DEBUG,
                       do_print=True, print_level=logging.INFO, print_format_type="simple", 
                       do_file=True, log_dir=LOG_DIR, file_level=logging.DEBUG, file_format_type="detail")
    return logger