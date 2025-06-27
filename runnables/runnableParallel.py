from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv


load_dotenv()

prompt1 = PromptTemplate(
    template= 'Generate a tweet about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template= 'Generate a Linkedin Post about {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()
model = ChatOpenAI()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic': 'ass'})

print(result['tweet'])
print(result['linkedin'])