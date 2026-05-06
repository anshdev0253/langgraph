from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat. Goodbye!")
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print('Bot: ', result.content)

print(chat_history)