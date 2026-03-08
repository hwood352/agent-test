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

3. Configure environment variables:
   
   a. Copy the sample environment file:
   ```bash
   cp .env.sample .env
   ```
   
   b. Edit the `.env` file and add your OpenAI API key:
   ```bash
   # Open .env in your preferred editor
   nano .env  # or vim, code, etc.
   ```
   
   c. Update the `OPENAI_API_KEY` value with your actual API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   
   **Note**: The `.env` file contains sensitive information and should never be committed to version control. Make sure it's listed in your `.gitignore` file.

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
├── simple_agent.py      # Main agent implementation with detailed comments
├── requirements.txt     # Python dependencies
├── .env.sample         # Sample environment configuration file
├── .env                # Your local environment variables (create from .env.sample)
└── README.md           # This file
```

## Environment Configuration

The project uses environment variables for configuration. All settings are defined in the `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-3.5-turbo` | No |
| `TEMPERATURE` | Model temperature (0.0-2.0) | `0.0` | No |
| `MAX_ITERATIONS` | Maximum agent iterations | `10` | No |
| `VERBOSE` | Show detailed agent reasoning | `true` | No |

See [`.env.sample`](.env.sample) for a complete example with detailed comments.

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
