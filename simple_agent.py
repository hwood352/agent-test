"""
Simple LangChain Agent Example

This script demonstrates a basic LangChain agent that can use tools
to answer questions and perform tasks.
"""

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# Define simple tools for the agent
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def string_length(text: str) -> str:
    """Returns the length of a string."""
    return f"The length of the text is: {len(text)} characters"


def reverse_string(text: str) -> str:
    """Reverses a string."""
    return f"Reversed text: {text[::-1]}"


# Create tools list
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for evaluating mathematical expressions. Input should be a valid Python expression like '2+2' or '10*5'."
    ),
    Tool(
        name="StringLength",
        func=string_length,
        description="Returns the length of a given string. Input should be the text you want to measure."
    ),
    Tool(
        name="ReverseString",
        func=reverse_string,
        description="Reverses a given string. Input should be the text you want to reverse."
    )
]


# Define the prompt template for the ReAct agent
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)


def create_simple_agent(api_key: str = None):
    """
    Creates and returns a simple LangChain agent.
    
    Args:
        api_key: OpenAI API key (optional, can be set via OPENAI_API_KEY env var)
    
    Returns:
        AgentExecutor: The configured agent executor
    """
    # Initialize the language model
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor


def main():
    """
    Main function to demonstrate the agent.
    """
    print("=" * 60)
    print("Simple LangChain Agent Demo")
    print("=" * 60)
    print("\nThis agent can use the following tools:")
    print("1. Calculator - for mathematical expressions")
    print("2. StringLength - to count characters in text")
    print("3. ReverseString - to reverse text")
    print("\n" + "=" * 60)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("\nNote: OPENAI_API_KEY environment variable not set.")
        print("To use this agent, set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nFor demonstration purposes, here's how the agent would work:")
        print("\nExample queries:")
        print("- 'What is 25 multiplied by 4?'")
        print("- 'How long is the word LangChain?'")
        print("- 'Reverse the word Hello'")
        return
    
    # Create the agent
    try:
        agent_executor = create_simple_agent()
        
        # Example queries
        example_queries = [
            "What is 15 + 27?",
            "How many characters are in the word 'LangChain'?",
            "Reverse the word 'Agent'"
        ]
        
        for query in example_queries:
            print(f"\n{'=' * 60}")
            print(f"Query: {query}")
            print("=" * 60)
            result = agent_executor.invoke({"input": query})
            print(f"\nFinal Answer: {result['output']}")
            
    except Exception as e:
        print(f"\nError running agent: {str(e)}")
        print("Make sure you have set a valid OPENAI_API_KEY")


if __name__ == "__main__":
    main()
