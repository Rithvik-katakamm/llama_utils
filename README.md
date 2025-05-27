 # OllamaChat

 **Enhanced conversational chat interface for local Ollama models in Python.**

 OllamaChat provides a flexible, session-based chat manager on top of Ollama's Python API. It offers:
 - Rich console output (via [rich]) with panels, tables, syntax highlighting, and spinners.
 - Session management: start, load, list, and save conversations as JSON.
 - Context injection: programmatically add text or file contexts to system prompts.
 - Interactive chat loop with commands: help, stats, search.

 ## Table of Contents
 - [Features](#features)
 - [Installation](#installation)
 - [Quickstart](#quickstart)
 - [Configuration](#configuration)
 - [Usage](#usage)
   - [Initialize](#initialize)
   - [Start a New Session](#start-a-new-session)
   - [Load an Existing Session](#load-an-existing-session)
   - [Add Context](#add-context)
   - [Send Messages](#send-messages)
   - [Interactive Chat](#interactive-chat)
   - [Session Stats and Search](#session-stats-and-search)
 - [API Reference](#api-reference)
 - [Conversation Storage](#conversation-storage)
 - [Requirements](#requirements)
 - [Contributing](#contributing)
 - [Support](#support)

 ## Features
 - **Model-Agnostic**: Works with any Ollama model (e.g., `deepseek-r1:7b`).
 - **Rich / Plain Output**: Auto-detects `rich` availability or falls back to plain text.
 - **Session Files**: Conversations stored as JSON in `conversations/<project>/`.
 - **Context Management**: Add text or file contents into the system prompt.
 - **Search & Stats**: Query past messages and view session statistics.
 - **Interactive Mode**: Built-in REPL with commands (`quit`, `help`, `stats`, `search`).

 ## Installation

 1. Ensure you have **Python 3.7+** installed.
 2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\\Scripts\\activate   # Windows
    ```
 3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

 ## Quickstart

 Below is a minimal example (e.g., in a Jupyter notebook or Python script):

 ```python
 from ollama_chat import OllamaChat

 # 1. Initialize with your model
 chat = OllamaChat(model="deepseek-r1:7b")

 # 2. Start a new session with a system prompt
 chat.start_new_session(
     system_prompt="You are a Python assistant. Provide clear explanations and code samples."
 )

 # 3. Send a message
 response = chat.send("How do I add a docstring to a function?")
 print("AI:", response)

 # 4. Inspect full history
 for msg in chat.messages:
     print(msg["role"], ":", msg["content"])
 ```

 ## Configuration

 - **model**: Ollama model name (e.g., `"gpt-4o-mini"`).
 - **visual_mode**: `"auto"`, `"rich"`, `"plain"`, or `"silent"`.
   - `auto`: uses rich if available and stdout is a TTY.
   - `silent`: suppresses all console output.
 - **project_name**: Organize sessions under `conversations/<project_name>/`.
 - **conversations_dir**: Base directory for session files (default: `conversations`).

 ## Usage

 ### Initialize
 ```python
 chat = OllamaChat(
     model="deepseek-r1:7b",
     conversations_dir="conversations",
     visual_mode="auto",
     project_name="my_project"
 )
 ```

 ### Start a New Session
 ```python
 chat.start_new_session(
     system_prompt="You are a helpful assistant."
 )
 # Saved as: conversations/my_project/YYYYMMDD_HHMMSS.json
 ```

 ### Load an Existing Session
 ```python
 sessions = chat.list_sessions()  # -> ['20230510_120000.json', ...]
 chat.load_session(sessions[0])
 ```

 ### Add Context
 ```python
 chat.add_context("Docstring Guidelines", "Use triple quotes for function docstrings.")
 # or from a file:
 chat.add_file_context("path/to/my_code.py")
 ```

 ### Send Messages
 ```python
 reply = chat.send("Show me an example docstring.", stream_output=False)
 print(reply)
 ```

 ### Interactive Chat
 ```python
 chat.start_interactive_chat()
 ```
 Built-in commands:
 - `quit`: Exit
 - `help`: Show commands
 - `stats`: Display message counts and characters
 - `search <query>`: Search past messages

 ### Session Stats and Search
 ```python
 stats = chat.get_session_stats()
 print(stats)

 results = chat.search_messages("docstring")
 for r in results:
     print(r["index"], r["role"], r["snippet"])
 ```

 ## API Reference

 #### class `OllamaChat(model, conversations_dir, visual_mode, project_name)`
 Constructor arguments:
 - `model` (str): Ollama model name.
 - `conversations_dir` (str): Directory for storing sessions.
 - `visual_mode` (str): `'auto'`, `'rich'`, `'plain'`, `'silent'`.
 - `project_name` (Optional[str]): Subfolder under conversations.

 #### `list_sessions(include_metadata=False) -> List[str]`
 List saved session filenames (JSON). Use `include_metadata=True` for details.

 #### `start_new_session(system_prompt=None, session_name=None)`
 Create a new session file and optionally send an initial system prompt.

 #### `load_session(filename) -> bool`
 Load messages and context from a saved session.

 #### `add_message(role, content, save=True) -> bool`
 Append a message. Roles: `'user'`, `'assistant'`, `'system'`.

 #### `add_context(title, content, context_type='text')`
 Add context metadata and inject as a system message.

 #### `add_file_context(filepath, title=None) -> bool`
 Load file contents as context.

 #### `send(user_input, stream_output=None) -> Optional[str]`
 Send a message to the model and receive a response.

 #### `start_interactive_chat()`
 Launch REPL loop with quick commands.

 #### `get_session_stats() -> Dict`
 Return counts of messages, characters, etc.

 #### `search_messages(query, role=None) -> List[Dict]`
 Search conversation history for matching content.

 #### `switch_to_project(project_name)`
 Change project folder and reset session state.

 ## Conversation Storage

 Sessions are saved under:
 ```
 conversations/<project_name>/<session_filename>.json
 ```
 Each JSON contains:
 ```json
 {
   "metadata": { model, project, created, last_modified, message_count, context_data },
   "messages": [ { role, content }, ... ]
 }
 ```

 ## Requirements

 - Python 3.7+
 - [ollama] Python package (`pip install ollama`)
 - [rich] for enhanced output (`pip install rich`) (optional)

 ## Contributing

 Contributions and issues are welcome! Feel free to open a pull request or issue.

 ## Support

 For questions or troubleshooting, check:
 - Your Ollama installation and `ollama` CLI (`ollama --help`)
 - `rich` availability for colored output
 - Issues on the project repository

 [ollama]: https://pypi.org/project/ollama
 [rich]: https://pypi.org/project/rich