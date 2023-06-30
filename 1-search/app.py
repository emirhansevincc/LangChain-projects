from langchain import OpenAI
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
import os

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


os.environ['OPENAI_API_KEY'] = 'YourAPIKey'

os.environ['SERPAPI_API_KEY'] = 'YourAPIKey'

llm = OpenAI(temperature=.9)

st.set_page_config(
    page_title="Project-3",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Project-3")

query = st.text_input("Enter Your Query Here :", placeholder="Enter a film category here")


name_template = '''
    % You are an AI bot that helps a user to give a famous film name about the topic below. You should give just one name and do not add anything, just the name of the film.
    Here is the topic: {topic}
'''

name_prompt_template = PromptTemplate(
    template=name_template,
    input_variables=['topic'],
)

script_template = '''
    % You are an AI bot that helps a user to write a script about the film below.
    Here is the title: {title}
'''

script_prompt_template = PromptTemplate(
    template=script_template,
    input_variables=['title'],
)

n_memory = ConversationBufferMemory(input_key='topic', memory_key='history')

s_memory = ConversationBufferMemory(input_key='title', memory_key='history')


# Let's create chains
name_chain = LLMChain(
    llm=llm,
    prompt=name_prompt_template,
    output_key='title',
    memory=n_memory,
)

script_chain = LLMChain(
    llm=llm,
    prompt=script_prompt_template,
    output_key='script',
    memory=s_memory,
)

seq_chain = SequentialChain(
    chains=[name_chain, script_chain],
    input_variables=['topic'],
    output_variables=['title', 'script'],
)

if query:

    res = seq_chain({'topic': query})

    st.write('AI is thinking about title for your query')
    st.write(res['title'])

    st.write('*****************************************')

    st.write('AI is thinking about script about the film')
    st.write(res['script'])

    # You can use WikipediaAPIWrapper instead of langchain agent ==> from langchain.utilities import WikipediaAPIWrapper
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )

    st.write('*****************************************')

    st.write(agent.run(f"What is the general information about {res['title']}?",))


    with st.expander('Name History'): 
        st.info(n_memory.buffer)

    with st.expander('Script History'): 
        st.info(s_memory.buffer)


"""
Here is the output of agent.run(f"What is the general information about {res['title']}?",)

Let's say our film is 'The Matrix'

> Entering new  chain...
I need to find out what the Matrix is

Action: Search
Action Input: The Matrix
Observation: Neo (Keanu Reeves) believes that Morpheus (Laurence Fishburne), an elusive figure considered to be the most dangerous man alive, can answer his question -- What is the Matrix? Neo is contacted by Trinity (Carrie-Anne Moss), a beautiful stranger who leads him into an underworld where he meets Morpheus. They fight a brutal battle for their lives against a cadre of viciously intelligent secret agents. It is a truth that could cost Neo something more precious than his life.… MORE

Thought: I need to find out more information
Action: Search
Action Input: The Matrix plot
Observation: Neo (Keanu Reeves) believes that Morpheus (Laurence Fishburne), an elusive figure considered to be the most dangerous man alive, can answer his question -- What is the Matrix? Neo is contacted by Trinity (Carrie-Anne Moss), a beautiful stranger who leads him into an underworld where he meets Morpheus. They fight a brutal battle for their lives against a cadre of viciously intelligent secret agents. It is a truth that could cost Neo something more precious than his life.

Thought: I now know the final answer
Final Answer: The Matrix is a science fiction action film written and directed by the Wachowskis. It stars Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. The plot follows a computer hacker named Neo who is contacted by the mysterious Morpheus and Trinity, who fight a secret war against a powerful computer program known as the Matrix. The Matrix was a box-office success and won four Academy Awards. It has since become one of the most influential science fiction films of all time.

> Finished chain.
"""