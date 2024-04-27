import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import time

from langchain_core.prompts import PromptTemplate


load_dotenv()

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)

@tool
def take_a_note(text):

    """
    To take a note
    """

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/Notes.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")
        return ("-note taken-")
    

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
def chat_history_recorder(text):
    """
    to record chat history
    """
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/chat_history.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")
        return ("-note taken-")

@tool
def chat_history_recall(text):
    """
    to recall chat history
    """

    file_path="C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/chat_history.txt"
    with open(file_path, "r") as file:
    # Read all lines from the file into a list
        lines = file.readlines()
    return (lines)



tools = [search_duck,wikipedia,take_a_note,chat_history_recorder,chat_history_recall]

tool_executor = ToolExecutor(tools)


class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# Define Ollama model
model_name = "dolphin-llama3"  # Replace with your desired Ollama model ollama run dolphin-llama3
# model_name = "phi3"    #ollama run phi3
# model_name = "phi3:instruct"    #ollama run phi3
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest" #ollama run cas/minicpm-3b-openhermes-2.5-v2:latest




model = ChatOllama(model=model_name,temperature=0)
# prompt = hub.pull("hwchase17/react")
# print ("prompt ",prompt)

# templete = """You are an AI agent that expert in following instructions. You will engage in conversation and provide relevant information to the best of your knowlaedge. You have a set of functions you can use.
# You have access to these funcitons:
# {tools}

# If user's input containts "search in internet" or similar and some sort of questions or query, take the action with "search_duck" tool.


# If user's input containts "stake a note" or similar and some sort of questions or query, take the action with "take_a_note" tool.


# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat 1 times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""



# templete = """Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do, if there is no need to use tool reply as Final answer
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat 2 times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""


# templete = """Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: Do I need to use a tool? Yes
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat 2 times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Thought: Do I need to use a tool? No
# Final Answer: [your response here]

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}"""

templete = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

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

Use chat_history_recorder tool for every human input and Final Answer.
And if user want to remember a thing use chat_history_recall for remembering things that was in conversation.


Begin!

Question: {input}
Thought:{agent_scratchpad}"""



prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools',], 
    # metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'react', 'lc_hub_commit_hash': 'd15fe3c426f1c4b3f37c9198853e4a86e20c425ca7f4752ec0c9b0e97ca7ea4d'} ,
    template=templete
)


agent_runnable = create_react_agent(model, tools, prompt)


def execute_tools(state):
    print("Called `execute_tools`")
    # Based on the continue condition
    # we know the last message involves a function call
    messages = [state["agent_outcome"]]
    last_message = messages[-1]

    tool_name = last_message.tool

    print(f"Calling tool: {tool_name}")

    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    # We call the tool_executor and get back a response

    response = tool_executor.invoke(action)

    return {"intermediate_steps": [(state["agent_outcome"], response)]}


def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    agent_outcome = agent_runnable.invoke(state)
    # print("agent_outcome : ",agent_outcome)
    state["chat_history"].append(agent_outcome)
    print("##################################")
    print("state: ",state)
    print("##################################")
    return {"agent_outcome": agent_outcome}


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    
    # If there is no function call, then we finish
    if "Action" not in last_message.log:
        return "end"
    # Otherwise if there is, we continue
    else:
        # return "end"
        return "continue"



# Define a new graph
workflow = StateGraph(AgentState)


# Define the two nodes we will cycle between
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END
    }
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.


workflow.add_edge("action", "agent")
app = workflow.compile()

# input_text = "Whats the current time?"
# input_text = "Can you make an internet search for Elon Musk?"
# input_text = "I wonder where was the latest F1 Race happened?"
# calculate 3+3

while True:
    input_text=input("Human: ")
    inputs = {"input": input_text, "chat_history": []}

    results = []
    
    # print(app.invoke(inputs))
    
    try:
        for s in app.stream(inputs):
            result = list(s.values())[0]
            results.append(result)
            print(result)
    except Exception as e:
        print("Error:", e)

