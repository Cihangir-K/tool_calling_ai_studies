from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferMemory
import time
from datetime import datetime

# prompt = hub.pull("hwchase17/react")

model_name = "dolphin-llama3"
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest" #ollama run cas/minicpm-3b-openhermes-2.5-v2:latest
# model_name = "stablelm-zephyr"


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

@tool
def wikipedia(text):
    """
    To make an wikipedia search 
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print("\n"+"WIKI Result :",wikipedia.run(text)+"\n")
    return wikipedia.run(text)

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)

@tool
def take_a_note(text):

    """
    To take a note.
    Use this format.
    Human: "Human input",
    AI: "AI response".
    """
    # use this format.
    # Human: "Human input",
    # AI: "AI response".

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/Notes_2.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")
        return ("-note taken-")

tools = [search_duck,get_now,take_a_note]

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
agent_executor = AgentExecutor(agent=agent,memory=memory, tools=tools,handle_parsing_errors=True,return_intermediate_steps=True,max_iterations=5)




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