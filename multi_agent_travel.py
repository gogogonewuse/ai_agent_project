import os
import streamlit as st
from agno.models.deepseek import DeepSeek
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.vectordb.lancedb import LanceDb,SearchType
from agno.knowledge.knowledge import Knowledge
#txt文本阅读器
from agno.knowledge.reader.text_reader import TextReader
# from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.embedder.cohere import CohereEmbedder
from agno.knowledge.reranker.cohere import CohereReranker
from agno.db.mysql import MySQLDb
#防止数据库断开
from sqlalchemy import create_engine
# =====================================================================
# 🌟 1. 网页基础配置（必须放在代码最顶部）
# =====================================================================
st.set_page_config(
    page_title="AI 智能路由协同系统",
    page_icon="✈️",
    layout="wide"
)
st.title("AI 智能旅游管家")
st.caption("当前架构：ai领队+美食专家+旅游专家")
if "messages" not in st.session_state:
    st.session_state.messages=[]
api_key = os.getenv("COHERE_API_KEY")  # 从环境变量读取

# =====================================================================
# 🌟 2. 使用 Streamlit 缓存机制，防止每次点网页按钮都重新加载数据库
# =====================================================================
@st.cache_resource
def init_core_agents():
    cohere_key=os.getenv("COHERE_API_KEY")
    deepseek_key=os.getenv("DEEPSEEK_API_KEY")
    serpapi_key=os.getenv("SERPAPI_API_KEY")
    # 配置共享大脑
    ai_model=DeepSeek(
    id="deepseek-v4-flash",
    api_key=deepseek_key
        )
    engine=create_engine(
        "mysql+pymysql://root:root@127.0.0.1:3306/ai",
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10
    )
    mysql_db=MySQLDb(
    db_engine=engine,
    db_url="mysql+pymysql://root:root@localhost:3306/chat_history",
    session_table="chat_history_01",
    
    )
    cohere_embedder=CohereEmbedder(id="embed-multilingual-v3.0",api_key=cohere_key,dimensions=1024)
    verctr_db=LanceDb(
    table_name="my_rerank_data",
    uri="lancedb_data",
    # embedder=SentenceTransformerEmbedder(id="BAAI/bge-small-zh-v1.5"),
    embedder=cohere_embedder,
    search_type=SearchType.vector,
    use_tantivy=False,
    reranker=CohereReranker(model="rerank-multilingual-v3.0",top_n=3,api_key=cohere_key)
    
    )
    my_serpapi=SerpApiTools(api_key=serpapi_key)
    knowledge_base=Knowledge(vector_db=verctr_db)
    knowledge_base.add_content(
    path="knowledge_docs",
    reader=TextReader(),
    upsert=True,
    )
    #美食专家
    food_agent=Agent(
    name="Food Expert",
    model=ai_model,
    tools=[my_serpapi],
    knowledge=knowledge_base,
    search_knowledge=True,
    description="你是一名美食专家，会根据用户的情况推荐吃什么样的美食",
    instructions=[
        "你只负责回答美食，餐厅，饭菜之类的问题",
        "先去数据库找数据，数据库没有的话可以使用tool工具去网上搜索"
    ],
    markdown=True
    )
    #出行管家
    travel_agent=Agent(
    name="travel Helper",
    model=ai_model,
    tools=[my_serpapi],
    description="你是一名旅游管家，专门负责查询天气，机票，酒店等试试联网信息。",
    instructions=[
        "只回答天气，订票，旅游，行程规划等相关问题，美食部分交给队友",
        "使用tool工具去网络上寻找最新的联网信息"
    ],
    markdown=True,

    )
    def call_food_expert(query:str)->str:
        response = food_agent.run(query)
        return response.content
    def call_travel_helper(query:str)->str:
        response=travel_agent.run(query)
        return response.content
        #创建核心领队 Agent（打开 debug 模式看交接过程）
    super_agent=Agent(
            name="team Pro",
            model=ai_model,
            tools=[call_food_expert,call_travel_helper],
            description="你是一名队伍领袖，专功与旅游行业，你的手下有两名专业的专家。",
            instructions=[
            "你的两名专家分别是美食专家和旅游专家,他们两位只负责自己的区域",
            "你只需要将问题解析，然后将不同的问题分发给你的手下.",
            "先在数据库或者一些本地文件查找数据，然后可以用tool等工具"
            ],
            db=mysql_db,
            add_history_to_context=True,
            session_id="multi_history_data_01",
            # knowledge=knowledge_base
        )
    return super_agent
super_my_agent=init_core_agents()
with st.sidebar:
    st.header("🔩控制台")
    session_id=st.text_input("当前会话ID",value="multi_history_web_01")
    if st.button("清除历史记录"):
        st.session_state.messages=[]
        st.rerun()
# if st.button("清除历史记录"):
#     st.session_state.messages=[]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
#流式交流
if user_input := st.chat_input("和你的智囊团旅游吧！"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})
    #动态渲染ai问题
    with st.chat_message("assistant"):
        #先创建一个空容器，然后一个一个塞字
        message_placeholder = st.empty()
        full_response=""
        with st.spinner("团队正在智能分发路由中..."):
            #核心改变：开启stream=true，此时返回的是一个迭代器
            response_stream=super_my_agent.run(user_input,session_id=session_id,stream=True)
            #循环每一个字
            for chunk in response_stream:
                if chunk.content:
                    full_response +=chunk.content
                    #试试刷新屏幕
                    message_placeholder.markdown(full_response+"▌")
            #吐字完毕，去掉打字光标
            message_placeholder.markdown(full_response)
    #最后完整的长文本加入到网页缓存里面
    st.session_state.messages.append({"role":"assistant","content":full_response})

#普通交流
# with st.sidebar:
#     st.header("🔩控制台")
#     session_id=st.text_input("当前会话ID",value="multi_history_web_01")
#     if st.button("清除历史记录"):
#         st.session_state.messages=[]
#         st.rerun()
# # if st.button("清除历史记录"):
# #     st.session_state.messages=[]
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if user_input := st.chat_input("和你的智囊团旅游吧！"):
#     with st.chat_message("user"):
#         st.markdown(user_input)
#     st.session_state.messages.append({"role":"user","content":user_input})
#     with st.chat_message("assistant"):
#         with st.spinner("团队正在智能路由分发中..."):
#             response=super_my_agent.run(user_input,session_id=session_id)
#             st.markdown(response.content)
#     st.session_state.messages.append({"role":"assistant","content":response.content})
#命令行交流
# print("你的小助手已启动！")
# print("-"*40)
# print("输入 “quit”或者'exit'退出 ")
# while True:
#     user_input=input("用户：")
#     if user_input.lower() in ['quit','exit']:
#         print("拜拜啦，下次见~👋")
#         break
#     if not user_input.strip():
#         continue
#     print("小助手正在全力思考中！！！💪")
#     response=super_agent.run(user_input)
#     print(f"\n小助手：{response.content}\n")
#     print("-"*40)