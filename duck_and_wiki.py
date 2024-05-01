from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

 
import os
import pprint 
from dotenv import load_dotenv
load_dotenv()

# env_var = os.environ 
# # Print the list of user's 
# print("User's Environment variable:") 
# pprint.pprint(dict(env_var), width = 1) 



def tavily_search(text):
    print()

    # api_key=os.getenv('TAVILY_API_KEY')
    api_key=os.environ['TAVILY_API_KEY']
    # print("api key: ",api_key)

    search= TavilySearchAPIWrapper(tavily_api_key=api_key)
    tool = TavilySearchResults(api_wrapper=search)

    result=tool.invoke({"query": text})

    print("\n"+"Tavily Search results: ",result+ "\n")
    return result

def search_duck(text):
    search = DuckDuckGoSearchRun()
    result =search.run(text)
    print("\n"+"DUCK DUCK results: ",result+ "\n")
    return result

def wikipedia(text):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    print("Wiki result ",wikipedia.run(text))
    return wikipedia.run(text)


# search_duck("seat belt safety test")

# wikipedia("wolverine")

tavily_search("latest tesla stock price")