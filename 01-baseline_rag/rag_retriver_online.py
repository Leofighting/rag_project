# -*- coding: utf-8 -*-
"""
@Time : 2026/9/15 9:51
@Author: janic
@File: rag_retriver_online.py
"""
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

from model import QwenLLM

langchain_llm = QwenLLM()

template = """
{our_text}
你能为上述内容创建一个包含{wordsCount}个词的推文吗？
"""

prompt = PromptTemplate(input_variables=["our_text", "wordsCount"],
                        template=template)

# prompt.format(our_text="我喜欢旅行，我已经去过6个国家。我计划不久后再去几个国家。",
#               wordsCount="3")

final_prompt = prompt.format(our_text="我喜欢旅行，我已经去过6个国家。我计划不久后再去几个国家。",
              wordsCount="3")

# print("=="*8)
# print("Prompt: ")
# print(final_prompt)
#
# print("=="*8)
# print("Answer: ")
# print(langchain_llm.invoke(final_prompt))

examples = [{'query': '什么是手机？',
             'answer': '手机是一种神奇的设备，可以装进口袋，就像一个迷你魔法游乐场。\
             它有游戏、视频和会说话的图片，但要小心，它也可能让大人变成屏幕时间的怪兽！'},
            {'query': '你的梦想是什么？',
             'answer': '我的梦想就像多彩的冒险，在那里我变成超级英雄，\
             拯救世界！我梦见欢笑声、冰淇淋派对，还有一只名叫Sparkles的宠物龙。'}]

example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=['query', "answer"],
    template=example_template
)

prefix = """你是一个5岁的小女孩，非常有趣、顽皮且可爱：
以下是一些例子：
"""

suffix = """
Question: {userInput}
Response: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['userInput'],
    example_separator='\n\n'
)

query1 = "月亮是什么？"
real_prompt = few_shot_prompt_template.format(userInput=query1)
# answer1 = langchain_llm.invoke(real_prompt)
#
# print("=="*22)
# print("Prompt: ")
# print(real_prompt)
#
# print("=="*22)
# print("Answer: ")
# print(answer1)

chain = few_shot_prompt_template | langchain_llm
# answer2 = chain.invoke({"userInput": "星星是什么？"})
# print(answer2)

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt_template_cls = PromptTemplate(
    template="Provide 5 example of {query}.\n{format_instructions}",
    input_variables=['query'],
    partial_variables={"format_instructions": format_instructions}
)

new_prompt = prompt_template_cls.format(query="太阳是什么？")
chain1 = prompt_template_cls | langchain_llm | CommaSeparatedListOutputParser()

answer3 = chain1.invoke({"query": "太阳是什么？"})
print(answer3)