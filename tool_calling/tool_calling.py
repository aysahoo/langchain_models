from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

llm_with_tools = llm.bind_tools([multiply])

messages = [HumanMessage('Can you multiply 3 with 40')]
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)

if ai_response.tool_calls:
    for tool_call in ai_response.tool_calls:
        if tool_call["name"] == "multiply":
            args = tool_call["args"]
            result = multiply.invoke(args)
            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_msg)
final_response = llm_with_tools.invoke(messages)
messages.append(final_response)

print(final_response.content)