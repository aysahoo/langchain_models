from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.0, max_completion_tokens=50)

result = model.invoke("Write a poem on cat")

print(result.content)
