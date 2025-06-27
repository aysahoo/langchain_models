from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough, RunnableParallel, RunnableLambda
from dotenv import load_dotenv


load_dotenv()

def count_words(text):
    return len(text.split(' '))

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a joke on the topc - {topic}',
    input_variables = ['topic']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'No. of words': RunnableLambda(count_words)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'ass'})

print(result['joke'])
print(result['No. of words'])