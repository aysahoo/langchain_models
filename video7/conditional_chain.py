from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['POSITIVE', 'NEGATIVE', 'NEUTRAL'] = Field(description='The sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback into positive, negative or neutral \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
prompt4 = PromptTemplate(
    template = 'Write an appropriate response to this neutral feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'POSITIVE', prompt2 | model | parser),
    (lambda x:x.sentiment == 'NEGATIVE', prompt3 | model | parser),
    (lambda x:x.sentiment == 'NEUTRAL', prompt4 | model | parser),
    RunnableLambda(lambda x: 'Couldn\'t find sentiment')
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'The product is ok okaish , not bad'})

chain.get_graph().print_ascii

print(result)
