#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/23 14:27:26
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''


from llm_eval.evaluator import SingleSelectionTask, do_evaluate    
from snippets import jdumps
from llm_eval.util import get_logger
logger = get_logger(__name__)



if __name__ == "__main__":
    task = SingleSelectionTask(
        name="ceval-glm66b-0123",
        data_path="./data/parsed/ceval/dev/*.jsonl",
        llm_config=dict(model_type="ZHIPU", model="chatglm_66b"),
        prompt_template='''请从ABCD四个【选项】中选出【问题】的答案,直接给出选项
【问题】：{question}

【选项】：{options}
答案是（不要推理过程！！输出只要包含A/B/C/D中的一个）：
''',
        extraction_pattern="(A|B|C|D)",
        infer_kwargs=dict(max_tokens=50, do_sample=False)  
    )
        
    detail,statistic = do_evaluate(task=task, max_num=None, work_num=8)
    logger.info(f"final result {jdumps(statistic)}")
    