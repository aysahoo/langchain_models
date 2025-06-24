from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant that can answer questions and help with tasks.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')

])
chat_history = []

with open ('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({'chat_history': chat_history, 'query':'What is the capital of India?'})