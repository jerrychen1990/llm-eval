#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/19 18:13:10
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''
from abc import abstractmethod
from enum import Enum
import os
import re
from typing import Type

from llm_eval.data_loader import TestItem, load_data, SelectionItem
from pydantic import BaseModel
from llm_eval.llm import LLM
from llm_eval.util import get_logger
from snippets.decorators import batch_process
from snippets.utils import groupby, dump2list


logger = get_logger(name=__name__)


class TaskType(str, Enum):
    SINGLE_SELECTION="SINGLE_SELECTION"
    
Task2Item = {
    TaskType.SINGLE_SELECTION:SelectionItem,
}
    

class Task(BaseModel):
    name :str
    data_path:str
    llm_config:dict
    prompt_template:str
    task_type:TaskType
    item_type:Type 
    infer_kwargs:dict = dict()
    
    @abstractmethod
    def evaluate_item(self, llm:LLM, item:TestItem)->dict:
        raise NotImplementedError
    
    
    
class SingleSelectionTask(Task):
    task_type:TaskType = TaskType.SINGLE_SELECTION
    item_type:Type = SelectionItem
    extraction_pattern:str
    
    def evaluate_item(self, item:SelectionItem, llm:LLM )->dict:
        prompt = self.prompt_template.format(**item.to_prompt_dict())
        resp = llm(prompt, **self.infer_kwargs)
        answers = re.findall(self.extraction_pattern, resp)
        if answers:
            model_answer = answers[0]
        else:
            logger.debug(f"cant extract answer from {resp}！") 
            model_answer = resp
        logger.debug(f"extracted model answer:{model_answer}")
        if item.answer:
            score = 1. if model_answer ==item.answer else 0.
        else:
            score = None
        rs_item = dict(**item.model_dump(), model_answer = model_answer, resp=resp, prompt=prompt, score=score)
        return rs_item
        
def do_evaluate(task:Task, work_num=1, max_num=None, tgt_dir:str=None):
    if not tgt_dir:
        tgt_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) + "/experiments"
    task_dir = os.path.join(tgt_dir, task.name)
    
    llm = LLM(**task.llm_config)
    data = load_data(task.data_path, task.item_type)
    if max_num:
        data = data[:max_num]    
    batch_func = batch_process(work_num=work_num, return_list=True)(task.evaluate_item)
    result = batch_func(data,llm=llm)
    result_dict = groupby(result, key=lambda x: os.path.basename(x["doc_path"]))
    for file_name, items in result_dict.items():
        tmp_path = os.path.join(task_dir, file_name)
        logger.info(f"dump {len(items)} items to {tmp_path}")
        dump2list(items, tmp_path)
    
    statistic_dict =dict()
    for k, v in result_dict.items():
        sum_score = sum(e["score"] for e in v if e["score"] is not None)
        total_num = len(v)
        score_num = len([e for e in v if e["score"] is not None])
        avg_score = sum_score / score_num if score_num > 0 else None
        statistic_dict[k] = dict(total_num=total_num, score_num=score_num, avg_score=avg_score, sum_score=sum_score)
    
    sum_score = sum(e["sum_score"] for e in statistic_dict.values())
    total_num = sum(e["total_num"] for e in statistic_dict.values())
    score_num = sum(e["score_num"] for e in statistic_dict.values())
    micro_avg_score = sum_score / score_num if score_num > 0 else None
    macro_avg_score = sum(e["avg_score"] for e in statistic_dict.values()) / len(statistic_dict)
    total_statistic = dict( total_num=total_num, score_num=score_num, 
                           micro_avg_score=micro_avg_score,macro_avg_score=macro_avg_score)
    statistic_dict["total"] = total_statistic
    dump2list(statistic_dict, os.path.join(task_dir, "result.json"))
    
    
    return result_dict,statistic_dict
        
   
    
if __name__ == "__main__":
    task = SingleSelectionTask(
        name="ceval",
        data_path="./data/parsed/ceval/dev/accountant_dev.jsonl",
        llm_config=dict(model_type="ZHIPU", model="glm-3-turbo"),
        prompt_template='''请从ABCD四个【选项】中选出【问题】的答案,直接给出选项
【问题】：{question}

【选项】：{options}
答案是（不要推理过程！！输出只要包含A/B/C/D中的一个）：
''',
        extraction_pattern="(A|B|C|D)",
        infer_kwargs=dict(max_tokens=50, do_sample=False)  
    )
        
    detail,statistic = do_evaluate(task=task, max_num=20, work_num=2)
    # logger.info(rs)
    
    
    
    
    
    
