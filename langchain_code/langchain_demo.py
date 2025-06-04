'''
构建本地知识库问答机器人:

如何从我们本地读取多个文档构建知识库，并且使用 Openai API 在知识库中进行搜索并给出答案
'''
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings
import os
import logging

os.environ['NO_PROXY'] = 'api.openai.rnd.huawei.com'
api_key = "sk-1234"
base_url = "http://api.openai.rnd.huawei.com/v1"

try:
    # 加载文件夹中的所有txt类型的文件
    loader = DirectoryLoader('D:\codes\llm-evaluation\langchain_code\data', glob='**/*.txt')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    print(f'documents:{len(documents)}')

    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)
    print(f'split_docs:{len(split_docs)}')

    # 初始化 openai 的 embeddings 对象
    embeddings = OllamaEmbeddings(base_url='http://localhost:11434', model='quentinz/bge-large-zh-v1.5:latest')
    # embeddings = OpenAIEmbeddings(model='nomic-embed-text', base_url=base_url, api_key=api_key)

    # 将 document 通过 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    docsearch = Chroma.from_documents(split_docs, embeddings)
    print(docsearch)

    llm = OpenAI(model_name="qwen2.5-72b-instruct",
                 api_key=api_key,
                 base_url=base_url)

    # 创建问答对象
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=docsearch.as_retriever(),
                                     return_source_documents=True)
    # 进行问答
    result = qa({"query": "完整的RAG应用流程主要包含哪两个阶段？"})
    print(result)
except Exception as e:
    logging.error(f'error:{e}')
