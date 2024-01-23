#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/23 14:27:59
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''

CEVAL_CONFIG = dict(data_path="./data/parsed/ceval/dev/*.jsonl",
                    llm_config=dict(model_type="ZHIPU", model="glm-3-turbo"),
                    prompt_template='''请从ABCD四个【选项】中选出【问题】的答案,直接给出选项
【问题】：{question}

【选项】：{options}
答案是（不要推理过程！！输出只要包含A/B/C/D中的一个）：
''',
                    extraction_pattern="(A|B|C|D)",
                    infer_kwargs=dict(max_tokens=50, do_sample=False))
