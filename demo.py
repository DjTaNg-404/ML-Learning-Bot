from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from ..demo.config import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, MODEL_NAME

# 定义数据模型
class Question(BaseModel):
    question: str = Field(description="具体的面试问题")
    primary_category: str = Field(description="一级分类")
    secondary_category: str = Field(description="二级分类") 
    difficulty: str = Field(description="难度等级")

class InterviewAnalysis(BaseModel):
    company: str = Field(description="公司名称")
    position: str = Field(description="面试岗位")
    questions: List[Question] = Field(description="面试问题列表")

# 创建解析器
parser = PydanticOutputParser(pydantic_object=InterviewAnalysis)

# 简化的提示模板 - 不需要手动指定JSON格式
analysis_prompt = ChatPromptTemplate.from_template("""
你是一个专业的面试经验分析师。请分析以下面经内容，提取关键信息并对面试问题进行分类。

面经内容：
{interview_text}

分类体系说明：
一级分类包括：编程语言基础、前端技术、后端框架、数据库技术、算法与数据结构、系统设计、大数据与AI、网络与系统、云服务与运维、项目经验

{format_instructions}

请开始分析：
""")

# 创建聊天模型
chat_model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    temperature=0.3,
    max_tokens=2000
)

# 创建链
chain = analysis_prompt | chat_model | parser

# 测试
test_interview = """
得物-算法平台-AI算法工程实习生 面经
积功德

职位描述：
1. 负责机器学习、深度学习等算法在得物业务场景的产品化工作
2. 包括但不限于如下方向：目标检测，图像分割，图像分类，NLP，多模态，大模型等
职位要求：
1. 熟悉Linux环境开发，熟练掌握 Python 语言，有较强的编码能力
2. 熟练使用一种深度学习框架如Pytorch、TensorFlow等，熟悉OpenCV、NumPy、Pandas等常用库
3. 对云原生有一定了解，有容器化使用经验者优先
4. 有GPU编程经验、熟悉算法模型部署、 TensorRT 优化工具者优先
5. 图像处理、模式识别、计算机视觉、计算机图形学、机器学习等计算机相关专业在读研究生优先

一面（2025.7.10）30min
HR发给我的邮件是上午 11 点，我 11 点进会议等了半个多小时没人退出去了，12 点多的时候，HR微信联系我说怎么没进飞书会议，然后我赶紧爬起来进会议。。。搞忘了，日本和国内有一个小时时差，麻了。。。

1. 面试官进来直接说你的简历我已经看过了，自我介绍一下吧
2. 几乎是纯聊天。。。面试官说我的经历非常匹配。。。
3. 大模型有没有推理优化经验？无，我说以前主要做CV算法，接触和使用过扩散模型。。。
4. 算法题：最大子数组和（秒了）
5. 硕士研究内容？
6. 偏向算法还是调度？有没有调度相关经验？无。。。
7. 写过CUDA吗？熟不熟？学校里深入学过，之后因为业务关系，没啥使用场景，可以再捡起来
8. 有没有NLP相关经验？基本的概念和算法比如 tf-idf, n-gram，word2vec 这些都是知道的，做过文本分类任务，了解 Transformer、CLIP
9. 有没有多卡推理优化经验？有多卡训练经验，多卡推理没做过。。。
10. 问什么时候能来实习？答最快这月底就能到岗，3个月时间可以保证，每周5天
11. 你知道岗位base地吗，能接受吗？我说就是期望在国内实习，上海完全能接受
12. 反问：组内主要业务场景？商品内容理解、文本理解、AI鉴定商品真伪、推理优化等

一面面试官貌似就是老大，结束后HR直接说过了，进offer流程。。。

今年暑期准备就去这个了，主要是面试官和善，面试体验好、务实，其余都是次要的（
没认真找，随便投投，本来想着回家吃饭睡觉的 日本饭好难吃啊。。。

"""

try:
    result = chain.invoke({
        "interview_text": test_interview,
        "format_instructions": parser.get_format_instructions()
    })
    print("分析结果：")
    print(f"公司：{result.company}")
    print(f"岗位：{result.position}")
    for i, q in enumerate(result.questions, 1):
        print(f"问题{i}：{q.question}")
        print(f"  分类：{q.primary_category} → {q.secondary_category}")
        print(f"  难度：{q.difficulty}")
        
except Exception as e:
    print(f"分析失败: {e}")