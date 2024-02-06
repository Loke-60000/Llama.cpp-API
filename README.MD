# Llama Chat Application

## Overview

A chat application using the Llama model via FastAPI for a REST API and a CLI for direct interaction.

## Features

- REST API with CORS enabled for web integration.
- CLI for interactive chatting in the terminal.
- Customizable chat settings (temperature, max tokens).

## Setup

1. **Requirements**: Python 3.8+, FastAPI, Uvicorn, `llama_cpp` package.
2. **Installation**: `pip install fastapi uvicorn`.
3. **Configuration**: Set `MODEL_PATH` in `api.py` and `main.py` to your Llama model location.

## Usage

- **API Server**: `uvicorn api:app --host 0.0.0.0 --port 8000`.
  - Visit `http://localhost:8000/docs` for API docs.
- **CLI**: Run `python main.py` and interact with the prompts
