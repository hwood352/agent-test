"""
Simple LangChain Agent with Custom Tools

This script demonstrates how to create a LangChain agent that can use multiple
tools to answer questions and perform tasks. The agent uses the ReAct (Reasoning
and Acting) framework to decide which tools to use based on the input query.

Author: LangChain Demo
License: MIT
"""

import os
from dotenv import load_dotenv
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ============================================================================
# Environment Configuration
# ============================================================================

# Load environment variables from .env file
# This allows us to keep sensitive information like API keys separate from code
load_dotenv()

# Retrieve the OpenAI API key from environment variables
# If not found, raise an error with helpful message
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file based on .env.sample and add your API key."
    )

# Get optional configuration with default values
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"


# ============================================================================
# Tool Definitions
# ============================================================================

def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    
    This tool uses Python's eval() function to calculate mathematical expressions.
    It's useful for performing arithmetic operations like addition, subtraction,
    multiplication, division, and more complex calculations.
    
    Args:
        expression (str): A mathematical expression as a string (e.g., "2 + 2", "10 * 5")
    
    Returns:
        str: The result of the calculation as a string
    
    Example:
        >>> calculator("25 * 4")
        "100"
    
    Note:
        In production, consider using a safer alternative to eval() like ast.literal_eval()
        or a dedicated math parser to prevent code injection vulnerabilities.
    """
    try:
        # Evaluate the mathematical expression
        result = eval(expression)
        return str(result)
    except Exception as e:
        # Return error message if evaluation fails
        return f"Error calculating expression: {str(e)}"


def string_length(text: str) -> str:
    """
    Counts and returns the number of characters in a given text string.
    
    This tool is useful for determining the length of strings, which can be
    helpful for validation, formatting, or analysis tasks.
    
    Args:
        text (str): The input text to measure
    
    Returns:
        str: The number of characters in the text as a string
    
    Example:
        >>> string_length("LangChain")
        "9"
    """
    try:
        # Calculate the length of the string
        length = len(text)
        return str(length)
    except Exception as e:
        # Return error message if operation fails
        return f"Error calculating string length: {str(e)}"


def reverse_string(text: str) -> str:
    """
    Reverses the order of characters in a given text string.
    
    This tool takes a string and returns it with all characters in reverse order.
    It's useful for text manipulation tasks or demonstrating string operations.
    
    Args:
        text (str): The input text to reverse
    
    Returns:
        str: The reversed text string
    
    Example:
        >>> reverse_string("Hello")
        "olleH"
    """
    try:
        # Reverse the string using Python's slice notation [::-1]
        reversed_text = text[::-1]
        return reversed_text
    except Exception as e:
        # Return error message if operation fails
        return f"Error reversing string: {str(e)}"


# ============================================================================
# Agent Setup
# ============================================================================

def create_agent():
    """
    Creates and configures a LangChain agent with custom tools.
    
    This function sets up:
    1. The language model (LLM) that powers the agent's reasoning
    2. A collection of tools the agent can use
    3. A prompt template that guides the agent's behavior
    4. The agent executor that runs the agent loop
    
    Returns:
        AgentExecutor: A configured agent ready to process queries
    """
    
    # Initialize the language model
    # ChatOpenAI is a wrapper around OpenAI's chat models (GPT-3.5, GPT-4, etc.)
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,  # Controls randomness (0 = deterministic, 2 = very random)
        openai_api_key=OPENAI_API_KEY
    )
    
    # Define the tools available to the agent
    # Each Tool needs a name, function, and description
    # The description is crucial - it tells the agent when to use each tool
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description=(
                "Useful for performing mathematical calculations. "
                "Input should be a valid mathematical expression like '2 + 2' or '10 * 5'. "
                "Returns the numerical result of the calculation."
            )
        ),
        Tool(
            name="StringLength",
            func=string_length,
            description=(
                "Useful for counting the number of characters in a text string. "
                "Input should be the text you want to measure. "
                "Returns the length as a number."
            )
        ),
        Tool(
            name="ReverseString",
            func=reverse_string,
            description=(
                "Useful for reversing the order of characters in a text string. "
                "Input should be the text you want to reverse. "
                "Returns the reversed text."
            )
        )
    ]
    
    # Create the ReAct prompt template
    # This template guides the agent through the reasoning and acting process
    # The agent will follow this pattern: Thought -> Action -> Action Input -> Observation
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

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
Thought: {agent_scratchpad}"""
    )
    
    # Create the ReAct agent
    # This agent will use the prompt template to decide which tools to use
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create the agent executor
    # This is what actually runs the agent loop, handling tool calls and responses
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=VERBOSE,  # If True, prints the agent's reasoning steps
        max_iterations=MAX_ITERATIONS,  # Prevents infinite loops
        handle_parsing_errors=True  # Gracefully handles parsing errors
    )
    
    return agent_executor


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function that demonstrates the agent's capabilities.
    
    This function:
    1. Creates the agent
    2. Runs several example queries to demonstrate different tools
    3. Prints the results
    """
    
    print("=" * 70)
    print("Simple LangChain Agent Demo")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {OPENAI_MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max Iterations: {MAX_ITERATIONS}")
    print(f"  Verbose: {VERBOSE}")
    print("=" * 70)
    
    # Create the agent
    agent_executor = create_agent()
    
    # Define example queries that demonstrate each tool
    example_queries = [
        "What is 25 multiplied by 4?",  # Will use Calculator tool
        "How long is the word LangChain?",  # Will use StringLength tool
        "Reverse the word Hello"  # Will use ReverseString tool
    ]
    
    # Run each example query
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Example {i}: {query}")
        print('=' * 70)
        
        try:
            # Invoke the agent with the query
            # The agent will reason about which tool to use and return a response
            response = agent_executor.invoke({"input": query})
            
            # Print the final answer
            print(f"\nFinal Answer: {response['output']}")
            
        except Exception as e:
            # Handle any errors that occur during execution
            print(f"\nError processing query: {str(e)}")
    
    print(f"\n{'=' * 70}")
    print("Demo completed!")
    print('=' * 70)


# Entry point of the script
# This ensures the main() function only runs when the script is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
