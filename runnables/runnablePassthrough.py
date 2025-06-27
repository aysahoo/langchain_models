from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel
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

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parrallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parrallel_chain)

result = final_chain.invoke({'topic': 'ass'})

print(result['joke'])
print(result['explanation'])
