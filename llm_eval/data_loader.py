#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/19 17:36:58
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''

import glob
import logging
from pydantic import BaseModel, Field
from typing import List, Optional

from tqdm import tqdm
from snippets.logs import getlog_detail
from snippets import load2list

logger = getlog_detail(name=__name__, level=logging.INFO)

class TextItem(BaseModel):
    def to_prompt_dict(self):
        raise NotImplementedError

class SelectionItem(TextItem):
    question:str
    answer:Optional[str]
    options:dict = Field(description="选项")
    doc_path:Optional[str] = Field(description="文件路径",default=None)
    label:Optional[str] 
    
    
    def to_prompt_dict(self):
        options = [f"{k}: {v}" for k,v in self.options.items()]
        options_str = "\n".join(options)
        return dict(question=self.question,options=options_str)
        
def load_data(glob_data_path:str, tgt_cls:type[TextItem])->List[TextItem]:
    data_paths = glob.glob(glob_data_path)
    rs_items = []
    for path in data_paths:
        logger.info(f"loading data from {path}")
        items = [tgt_cls(**e,doc_path=path) for e in tqdm(load2list(path))]
        rs_items.extend(items)
    logger.info(f"load data done, loaded {len(rs_items)} items")
    return rs_items



if __name__ == "__main__":
    items = load_data("./data/parsed/ceval/**/*.jsonl", SelectionItem)
    
    
        
        
        
        
        
        
        
    
    
    