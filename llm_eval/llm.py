#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/19 17:33:25
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''

import logging
from snippets.logs import getlog_detail

logger = getlog_detail(name=__name__, level=logging.INFO)

class LLM:
    def __init__(self, model_type:str, model:str):
        self.model_type = model_type
        self.model = model

    def __call__(self, prompt:str, **kwargs) -> str:
        if self.model_type.upper()=="ZHIPU":
            from agit.backend.zhipuai_bk import call_llm_api
            resp = call_llm_api(prompt=prompt, model=self.model, history=[], stream=False, logger=logger, **kwargs)
            return resp
        raise ValueError(f"invalid model_type:{self.model_type}")