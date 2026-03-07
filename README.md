# Simple LangChain Agent

A simple demonstration of a LangChain agent with basic tools.

## Overview

This project demonstrates a basic LangChain agent that can use multiple tools to answer questions and perform tasks. The agent uses the ReAct (Reasoning and Acting) framework to decide which tools to use based on the input query.

## Features

The agent includes three simple tools:

1. **Calculator** - Evaluates mathematical expressions
2. **StringLength** - Counts the number of characters in a text
3. **ReverseString** - Reverses a given string

## Installation

1. Clone this repository:
```bash
git clone https://github.com/hwood352/agent-test.git
cd agent-test
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

Run the agent:
```bash
python simple_agent.py
```

The script will demonstrate the agent by running example queries that use different tools.

## Example Queries

- "What is 25 multiplied by 4?"
- "How long is the word LangChain?"
- "Reverse the word Hello"

## Requirements

- Python 3.8+
- LangChain
- OpenAI API key

## Project Structure

```
agent-test/
├── simple_agent.py      # Main agent implementation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

The agent uses the ReAct framework which follows this pattern:

1. **Question**: Receives an input question
2. **Thought**: Reasons about what to do
3. **Action**: Selects a tool to use
4. **Action Input**: Provides input to the tool
5. **Observation**: Receives the tool's output
6. **Repeat** steps 2-5 as needed
7. **Final Answer**: Provides the final response

## License

MIT License
