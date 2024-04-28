from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain.agents import load_tools
from datetime import datetime
import time

# search_tool = DuckDuckGoSearchRun()
@tool('DuckDuckGoSearch')
def search_tool(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

@tool
def wikipedia(text):
    """
    To make an wikipedia search 
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print("\n"+"WIKI Result :",wikipedia.run(text)+"\n")
    return wikipedia.run(text)

@tool('save_to_note')
def Save_to_note(text):
    """
    To Save summary info to a note
    """
    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    with open("C:/Users/E.C.E/anaconda3/envs/tool_calling_ai/ck/Notes.txt", "a") as f:
        f.write(time_string+"\n" + "\n" + text + "\n" + "\n")
        print("Note taken!")
        return ("-note taken-")

# Loading Human Tools
human_tools = load_tools(["human"])


# Define Ollama model
model_name = "dolphin-llama3"  # Replace with your desired Ollama model ollama run dolphin-llama3
# model_name ="cas/minicpm-3b-openhermes-2.5-v2:latest" #ollama run cas/minicpm-3b-openhermes-2.5-v2:latest

temperature=0.2

# To Load Local models through Ollama
llm_model = Ollama(model=model_name)


Human_goal=input("your goal: ")

po = Agent(
    role="Product Owner",
    goal=Human_goal,
    backstory="""As the Product Owner you are responsable to give best result. 
	""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model
)

researcher = Agent(
    role="Researcher",
    goal="search in internet to find the best information",
    backstory="""You are a master of internet searching, with a profound knowledge DuckDuckGoSearchRun.
    For search, correct format is giving the search_query as string.
    """,
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
    max_iter=5,
    memory=True,
    tools=[search_tool]+human_tools,
)

reviewver = Agent(
    role="Reviewer",
    goal="Review the data to find correct information that user asked",
    backstory="""You are a guardian of data quality, with a sharp eye for detail in correct information.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
)

writer = Agent(
    role="writer",
    goal="write summarized data to file.",
    backstory="""You are a writer.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model,
    tools=[Save_to_note]
)

task1 = Task(
    description=f"""Conduct a comprehensive analysis of informaiton determined by the human. ASK THE HUMAN for the area of interest.\n
        The current time is {datetime.now()}.
        This tool aims to enhance operational efficiency and reliability. 
        Focus on to find the correct information. 
        Ensure that each information is detailed, specifying the context, the user's goal, and the desired outcome, 
        to guide the researcher team in creating a solution that meets users' needs.
    """,
    expected_output="a title and a definition of done",
    agent=po,
)

task2 = Task(
    description="""search in internet, and gather information. 
    """,
    expected_output='A comprehensive full report of the specified human topic, leave nothing out',
    agent=researcher,
)

task3 = Task(
    description="""Create a summary about info. 
    """,
    expected_output="A Summary about info",
    agent=reviewver,
)
task4 = Task(
    description="""Save the summary of data to notes.txt file
    """,
    expected_output="A Summary about info",
    agent=writer,
)

crew = Crew(
    agents=[po, researcher, reviewver, writer],
    tasks=[task1, task2, task3, task4],
    verbose=2,
    process=Process.sequential,
)

result = crew.kickoff()

print("######################")
print(result)