from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
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

@tool
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



# To Load Local models through Ollama
llm_model = Ollama(model=model_name)
coding_model = Ollama(model=model_name)

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
    For search, orrect format is {'search_query': 'Your query here'}.
    """,
    verbose=True,
    allow_delegation=False,
    llm=coding_model,
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
    description="""Ensure the quality of the information. And Save the data to notes.txt file
    """,
    expected_output="A Summary about info",
    agent=reviewver,
)

crew = Crew(
    agents=[po, researcher, reviewver],
    tasks=[task1, task2, task3],
    verbose=2,
    process=Process.sequential,
)

result = crew.kickoff()

print("######################")
print(result)