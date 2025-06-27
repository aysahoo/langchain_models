from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI()

prompt = PromptTemplate(
    template = 'Write a summary of the pdf \n {pdf}',
    input_variables=['pdf']
)

parser = StrOutputParser()

loader = PyPDFLoader('RAG/docLoaders/Sequence to Sequence Learning with NN.pdf')

docs =loader.load()

chain = prompt | model | parser

print(chain.invoke({'pdf' : docs[0].page_content}))
