from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a joke on the topc - {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a short and simple explanation from the following text \n {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

model = ChatOpenAI()

chain = RunnableSequence( prompt1, model, parser)

text = chain.invoke({'topic': 'ass'})
print(text)

chain2 = RunnableSequence( prompt2, model, parser)

print(chain2.invoke({'text': text}))
