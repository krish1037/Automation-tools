from langchain_core.tools import tool # for mading the tool of using the function
from langchain.agents import AgentType , initialize_agent , tool # for makin the ajent of the AI
from langchain_google_genai import ChatGoogleGenerativeAI# loading the language of the google model 
from langchain.memory import ConversationBufferMemory# it give the ability to remember the privious steps which has been executed
import os
os.environ['GOOGLE_API_KEY'] = 'Enter your own API key'
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',temperature=0.2)# specifing the model to be used in the llm 
@tool# this is the decorator which is used to enhace the functionality of the already existing function now here the function name is now used as the tool function which exist in the function name tools
def idea(idea_text:str)->str:
    """ Refine the idea of the project which has been given by the user ."""# this sring is written to tell the user of the function that what will this function do in short
    prompt=f"Refine the Rough idea of the project specific ,innovative and unique :\n\n{idea_text}"
    return llm.invoke(prompt)# this is used for the telling the llm model that take the prompt and analysis the info
@tool
def start_up(idea_text:str)->str:
    """ This is to find out the is that idea is physible and convertable to the start up"""
    prompt=f"Tell me on based of the idea of the project I had can it been physible and convertable into start-up:\n\n{idea_text} "
    return llm.invoke(prompt)
@tool
def business_model(idea_text:str)->str:
    """This is to find our the revenue and business model of the startup idea"""
    prompt=f"Tell how can this start up idea can generate the revenue and do the bussines and stay in the market :\n\n{idea_text}"
    return llm.invoke(prompt)
@tool
def user(idea_text:str)->str:
    """ It tells you who will be the user of this application """
    prompt=f"Find and gave the information who will be the user of this application :\n\n{idea_text}" 
    return llm.invoke(prompt)
tools =[idea,start_up,business_model,user]# isme tools  ko add karna  he  like chainning of the tools use of the multiple tools into one automation task
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(tools,llm,agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory,verbose=True)
agent.run("Refine my idea, analyze startup feasibility, suggest a business model, and identify users for: A project that detects whether a social media post or article is true or misleading ")
