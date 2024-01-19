#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/19 18:13:10
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''
from enum import Enum
import re

from tqdm import tqdm
from data_loader import load_data, SelectionItem
from pydantic import BaseModel
from llm_eval.llm import LLM
import logging
from snippets.logs import getlog_detail


logger = getlog_detail(name=__name__, level=logging.INFO)


class TaskType(str, Enum):
    SINGLE_SELECTION="SINGLE_SELECTION"
    
Task2Item = {
    TaskType.SINGLE_SELECTION:SelectionItem,
}
    

class Task(BaseModel):
    data_path:str
    task_type:TaskType
    llm_config:dict
    prompt_template:str
    extraction_pattern:str
    infer_kwargs:dict = dict()
    
    
    

    

def do_evaluate(task:Task):
    llm = LLM(**task.llm_config)
    data = load_data(task.data_path, Task2Item[task.task_type])
    for item in tqdm(data):
        prompt = task.prompt_template.format(**item.to_prompt_dict())
        resp = llm(prompt, **task.infer_kwargs)
        answer = re.findall(task.extraction_pattern, resp)[0]
        logger.info(f"get {resp}, {answer=} with {prompt=}")
        break
        
    
    
    
    
    
    
    
if __name__ == "__main__":
    task = Task(
        data_path="./data/parsed/ceval/test/accountant_test.jsonl",
        task_type=TaskType.SINGLE_SELECTION,
        llm_config=dict(model_type="ZHIPU", model="chatglm_6b"),
        prompt_template='''请从ABCD四个【选项】中选出【问题】的答案
        【选项】：{options}
        【问题】：{question}
        ''',
        extraction_pattern="A|B|C|D",
        infer_kwargs=dict(max_tokens=50, do_sample=False)  
    )
        
    do_evaluate(task=task)
    
    
    
    
    
    
    


def main(data_path:str, model_config:dict):
    data = load_data(data_path)
    
    
    pass
