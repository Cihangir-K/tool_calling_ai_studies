from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory


# prompt = hub.pull("hwchase17/react")

model_name = "dolphin-llama3"
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest" #ollama run cas/minicpm-3b-openhermes-2.5-v2:latest

model = ChatOllama(model=model_name,temperature=0)


@tool
def search_duck(text):
    """
    To make an internet search 
    """
    search = DuckDuckGoSearchRun()
    result =search.run(text)
    print("\n"+"DUCK DUCK results: ",result+ "\n")
    return result

tools = [search_duck]

templete = """ You have a chat_history. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

{chat_history}
Question: the input question you must answer

Thought: Do I need to use a tool? If answer No
Final Answer: [your response here]

Thought: Do I need to use a tool? If answer Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat 2 times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question



Begin!

Question: {input}
Thought:{agent_scratchpad}"""



prompt = PromptTemplate(
    input_variables=['agent_scratchpad', "chat_history",'input', 'tool_names', 'tools',], 
    template=templete
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent,memory=memory, tools=tools,handle_parsing_errors=True,return_intermediate_steps=True,max_iterations=2)




# Use with chat history
from langchain_core.messages import AIMessage, HumanMessage
while True:

    input_text=input("input: ")
    result=agent_executor.invoke(
        {
            "input": input_text,
            # Notice that chat_history is a string
            # since this prompt is aimed at LLMs, not chat models
            "chat_history": "Human: My name is Bob"
            "AI: Hello Bob!",
                    }
                )
    
    print("AI:",result)