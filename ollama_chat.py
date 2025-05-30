import ollama
import sys
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Union
import time
from transformers import AutoTokenizer

# Rich library for enhanced console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for enhanced visual experience: pip install rich")

CONVERSATIONS_DIR = "conversations"

class OllamaChat:
    """
    Enhanced conversational chat sessions with Ollama models.
    Features rich console interface, context management, and Jupyter integration.
    """

    def __init__(self, model='deepseek-r1:7b', conversations_dir=CONVERSATIONS_DIR, 
                 visual_mode='auto', project_name=None):
        """
        Initialize the enhanced chat manager.

        Args:
            model (str): The Ollama model name
            conversations_dir (str): Directory to store chat files
            visual_mode (str): 'auto', 'rich', 'plain', 'silent'
            project_name (str): Optional project name for session organization
        """
        self.model = model
        self.conversations_dir = conversations_dir
        self.messages = []
        self.filepath = None
        self.project_name = project_name
        self.context_data = []
        
        # Visual mode setup
        self.visual_mode = visual_mode
        if visual_mode == 'auto':
            self.visual_mode = 'rich' if RICH_AVAILABLE and sys.stdout.isatty() else 'plain'
        
        self.console = Console() if RICH_AVAILABLE and self.visual_mode == 'rich' else None
        self._ensure_conversations_dir()
        
        if self.visual_mode != 'silent':
            self._print_welcome()

    def _print_welcome(self):
        """Print welcome message based on visual mode."""
        if self.console:
            panel = Panel(
                f"[bold green]🤖 OllamaChat Enhanced v2.0[/bold green]\n"
                f"[blue]Model:[/blue] {self.model}\n"
                f"[yellow]Project:[/yellow] {self.project_name or 'Default'}\n"
                f"[cyan]Sessions:[/cyan] {len(self.list_sessions())}",
                title="[bold]Welcome[/bold]",
                border_style="green"
            )
            self.console.print(panel)
        else:
            print(f"--- OllamaChat Enhanced v2.0 (Model: {self.model}) ---")
            if self.project_name:
                print(f"Project: {self.project_name}")

    def _ensure_conversations_dir(self):
        """Ensure conversation storage directory exists."""
        project_dir = os.path.join(self.conversations_dir, self.project_name or 'default')
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
        self.project_dir = project_dir

    def _generate_filepath(self, custom_name=None):
        """Generate filepath for new session."""
        if custom_name:
            filename = f"{custom_name}.json"
        else:
            now = datetime.now()
            filename = f"{now.strftime('%Y%m%d_%H%M%S')}.json"
        return os.path.join(self.project_dir, filename)

    def _save(self):
        """Save current conversation with metadata."""
        if not self.filepath:
            return

        metadata = {
            "model": self.model,
            "project": self.project_name,
            "created": getattr(self, 'created_time', datetime.now().isoformat()),
            "last_modified": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "context_data": self.context_data
        }
        
        data_to_save = {
            "metadata": metadata,
            "messages": self.messages
        }
        
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self._print_error(f"Could not save conversation: {e}")

    def _print_message(self, role, content, stream=False):
        """Print message with appropriate formatting."""
        if self.visual_mode == 'silent':
            return
            
        if self.console:
            colors = {'user': 'blue', 'assistant': 'green', 'system': 'yellow'}
            icons = {'user': '👤', 'assistant': '🤖', 'system': '🔧'}
            
            if not stream:
                panel = Panel(
                    content,
                    title=f"[bold]{icons.get(role, '•')} {role.title()}[/bold]",
                    border_style=colors.get(role, 'white'),
                    title_align="left"
                )
                self.console.print(panel)
            else:
                self.console.print(f"[{colors.get(role, 'white')}]{icons.get(role, '•')} {role.title()}:[/{colors.get(role, 'white')}] ", end="")
        else:
            print(f"\n{role.title()}: {content}")

    def _print_error(self, message):
        """Print error message."""
        if self.visual_mode == 'silent':
            return
            
        if self.console:
            self.console.print(f"[bold red]❌ Error:[/bold red] {message}")
        else:
            print(f"[Error] {message}")

    def _print_success(self, message):
        """Print success message."""
        if self.visual_mode == 'silent':
            return
            
        if self.console:
            self.console.print(f"[bold green]✅ Success:[/bold green] {message}")
        else:
            print(f"[Success] {message}")

    # Core functionality (enhanced versions of your original methods)
    
    def list_sessions(self, include_metadata=False):
        """List saved conversation files with optional metadata."""
        files = [f for f in os.listdir(self.project_dir) if f.endswith('.json')]
        
        if not include_metadata:
            return sorted(files, reverse=True)
        
        sessions_with_meta = []
        for filename in files:
            filepath = os.path.join(self.project_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    meta = data.get('metadata', {})
                    sessions_with_meta.append({
                        'filename': filename,
                        'metadata': meta,
                        'preview': self._get_session_preview(data.get('messages', []))
                    })
            except:
                sessions_with_meta.append({
                    'filename': filename,
                    'metadata': {},
                    'preview': "Unable to load preview"
                })
        
        return sorted(sessions_with_meta, key=lambda x: x['metadata'].get('last_modified', ''), reverse=True)

    def _get_session_preview(self, messages):
        """Get preview of last message in session."""
        if not messages:
            return "Empty session"
        
        last_msg = messages[-1]
        content = last_msg.get('content', '')
        preview = content[:50] + '...' if len(content) > 50 else content
        return f"Last: \"{preview}\""

    def load_session(self, filename):
        """Load conversation with enhanced error handling."""
        self.filepath = os.path.join(self.project_dir, filename)
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.messages = data.get('messages', [])
                self.context_data = data.get('metadata', {}).get('context_data', [])
                
                metadata = data.get('metadata', {})
                loaded_model = metadata.get('model', 'N/A')
                
                if self.visual_mode != 'silent':
                    self._print_success(f"Loaded {filename}")
                    if loaded_model != self.model:
                        print(f"[Warning] Session used {loaded_model}, now using {self.model}")
                
                return True
        except Exception as e:
            self._print_error(f"Could not load {filename}: {e}")
            self.messages = []
            self.context_data = []
            self.filepath = None
            return False

    def start_new_session(self, system_prompt=None, session_name=None):
        """Start new session with optional custom name."""
        self.filepath = self._generate_filepath(session_name)
        self.messages = []
        self.context_data = []
        self.created_time = datetime.now().isoformat()
        
        if system_prompt:
            self.add_message('system', system_prompt, save=False)
        
        self._save()
        
        if self.visual_mode != 'silent':
            session_file = os.path.basename(self.filepath)
            self._print_success(f"Started new session: {session_file}")

    # New enhanced methods

    def add_context(self, title, content, context_type='text'):
        """Add context data to the conversation."""
        context_item = {
            'title': title,
            'content': content,
            'type': context_type,
            'added_at': datetime.now().isoformat()
        }
        self.context_data.append(context_item)
        
        # Add to system context
        context_msg = f"Context - {title}:\n{content}"
        self.add_message('system', context_msg, save=True)
        
        if self.visual_mode != 'silent':
            self._print_success(f"Added context: {title}")

    def add_file_context(self, filepath, title=None):
        """Add file content as context."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_title = title or f"File: {os.path.basename(filepath)}"
            self.add_context(file_title, content, 'file')
            return True
        except Exception as e:
            self._print_error(f"Could not read file {filepath}: {e}")
            return False

    def extract_code_blocks(self, text=None):
        """Extract code blocks from text or last AI response."""
        if text is None:
            if not self.messages or self.messages[-1]['role'] != 'assistant':
                return []
            text = self.messages[-1]['content']
        
        # Find code blocks marked with ```
        pattern = r'```(\w+)?\n?(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        code_blocks = []
        for lang, code in matches:
            code_blocks.append({
                'language': lang.strip() if lang else 'text',
                'code': code.strip()
            })
        
        return code_blocks

    def get_last_response(self):
        """Get the last AI response."""
        for msg in reversed(self.messages):
            if msg['role'] == 'assistant':
                return msg['content']
        return None

    def search_messages(self, query, role=None):
        """Search messages for specific content."""
        results = []
        for i, msg in enumerate(self.messages):
            if role and msg['role'] != role:
                continue
            if query.lower() in msg['content'].lower():
                results.append({
                    'index': i,
                    'role': msg['role'],
                    'content': msg['content'],
                    'snippet': self._get_search_snippet(msg['content'], query)
                })
        return results

    def _get_search_snippet(self, text, query, context_chars=100):
        """Get snippet around search query."""
        query_pos = text.lower().find(query.lower())
        if query_pos == -1:
            return text[:100] + '...'
        
        start = max(0, query_pos - context_chars // 2)
        end = min(len(text), query_pos + context_chars // 2)
        
        snippet = text[start:end]
        if start > 0:
            snippet = '...' + snippet
        if end < len(text):
            snippet = snippet + '...'
        
        return snippet

    def switch_to_project(self, project_name):
        """Switch to different project context."""
        self.project_name = project_name
        self._ensure_conversations_dir()
        self.messages = []
        self.context_data = []
        self.filepath = None
        
        if self.visual_mode != 'silent':
            self._print_success(f"Switched to project: {project_name}")

    def get_session_stats(self):
        """Get statistics about current session."""
        if not self.messages:
            return {}
        
        stats = {
            'total_messages': len(self.messages),
            'user_messages': len([m for m in self.messages if m['role'] == 'user']),
            'assistant_messages': len([m for m in self.messages if m['role'] == 'assistant']),
            'system_messages': len([m for m in self.messages if m['role'] == 'system']),
            'total_characters': sum(len(m['content']) for m in self.messages),
            'context_items': len(self.context_data)
        }
        
        return stats

    # Enhanced core methods
    
    def add_message(self, role, content, save=True):
        """Add message with validation and formatting."""
        if role not in ['user', 'assistant', 'system']:
            self._print_error(f"Invalid role: {role}")
            return False
            
        self.messages.append({'role': role, 'content': content})
        
        if save and self.filepath:
            self._save()
        
        return True

    def send(self, user_input, stream_output=None):
        """Enhanced send with better streaming and error handling."""
        if not self.filepath:
            self._print_error("No active session. Use start_new_session() first.")
            return None

        # Auto-detect streaming preference
        if stream_output is None:
            stream_output = self.visual_mode in ['rich', 'plain']

        self.messages.append({'role': 'user', 'content': user_input})
        
        if stream_output and self.visual_mode != 'silent':
            self._print_message('user', user_input)

        full_response = ""
        
        try:
            if stream_output and self.console:
                # Rich streaming with progress
                with self.console.status("[green]🤖 AI is thinking...", spinner="dots"):
                    time.sleep(0.5)  # Brief pause for effect
                
                self._print_message('assistant', '', stream=True)
            
            stream = ollama.chat(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

            for chunk in stream:
                part = chunk['message']['content']
                if stream_output and self.visual_mode != 'silent':
                    if self.console:
                        self.console.print(part, end='', style="green")
                    else:
                        print(part, end='', flush=True)
                full_response += part
            
            if stream_output and self.visual_mode != 'silent':
                print()  # Newline after streaming

            self.messages.append({'role': 'assistant', 'content': full_response})
            self._save()
            return full_response

        except ollama.ResponseError as e:
            self._print_error(f"Ollama error: {e.error}")
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            return None
        except Exception as e:
            self._print_error(f"Unexpected error: {e}")
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            return None

    # Interactive and display methods
    
    def display_session_selector(self):
        """Display interactive session selector."""
        sessions = self.list_sessions(include_metadata=True)
        
        if self.console:
            table = Table(title="📂 Your Conversations", show_header=True)
            table.add_column("#", style="cyan", width=3)
            table.add_column("Session Name", style="white")
            table.add_column("Preview", style="dim")
            table.add_column("Modified", style="yellow")
            
            table.add_row("N", "[bold green]🆕 Start New Chat[/bold green]", "", "")
            
            for i, session in enumerate(sessions[:10], 1):  # Show max 10 recent
                name = session['filename'].replace('.json', '')
                preview = session['preview']
                modified = session['metadata'].get('last_modified', 'Unknown')
                if modified != 'Unknown':
                    modified = datetime.fromisoformat(modified).strftime('%m/%d %H:%M')
                
                table.add_row(str(i), name, preview, modified)
            
            self.console.print(table)
        else:
            print("\n--- Session Selector ---")
            print(" [N] Start New Chat")
            for i, session in enumerate(sessions[:10], 1):
                name = session['filename'].replace('.json', '')
                preview = session['preview']
                print(f" [{i}] {name}")
                print(f"     {preview}")

    def start_interactive_chat(self):
        """Start enhanced interactive chat loop."""
        if not self.filepath:
            self._print_error("No active session. Use start_new_session() first.")
            return
        
        if self.visual_mode != 'silent':
            if self.console:
                self.console.print("\n[bold cyan]💬 Interactive Chat Started[/bold cyan]")
                self.console.print("[dim]Type 'quit' to exit, 'help' for commands[/dim]\n")
            else:
                print("\n--- Interactive Chat Started ---")
                print("Type 'quit' to exit, 'help' for commands\n")
        
        # Show recent history
        if len(self.messages) > 1:
            self._display_recent_history()

        while True:
            try:
                if self.console:
                    user_input = Prompt.ask("[blue]👤 You")
                else:
                    user_input = input("\nYou: ")
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                elif user_input.lower().startswith('search '):
                    query = user_input[7:]
                    self._show_search_results(query)
                    continue
                elif not user_input.strip():
                    continue
                
                self.send(user_input, stream_output=True)
                
            except (KeyboardInterrupt, EOFError):
                break
        
        if self.visual_mode != 'silent':
            self._print_success("Chat session ended")

    def _display_recent_history(self, limit=3):
        """Display recent conversation history."""
        if self.visual_mode == 'silent':
            return
            
        recent_messages = self.messages[-limit*2:] if len(self.messages) > limit*2 else self.messages
        
        if self.console:
            self.console.print("[dim]--- Recent History ---[/dim]")
            for msg in recent_messages:
                content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                self._print_message(msg['role'], content)
            self.console.print("[dim]--- End History ---[/dim]\n")
        else:
            print("--- Recent History ---")
            for msg in recent_messages:
                content = msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                print(f"{msg['role'].title()}: {content}")
            print("--- End History ---\n")

    def _show_help(self):
        """Show available commands."""
        if self.console:
            help_text = """
[bold]Available Commands:[/bold]
• [cyan]quit[/cyan] - Exit the chat
• [cyan]help[/cyan] - Show this help message
• [cyan]stats[/cyan] - Show session statistics
• [cyan]search <query>[/cyan] - Search messages for text
• Just type your message to chat!
            """
            self.console.print(Panel(help_text, title="Help", border_style="blue"))
        else:
            print("\nAvailable Commands:")
            print("• quit - Exit the chat")
            print("• help - Show this help message")
            print("• stats - Show session statistics")
            print("• search <query> - Search messages")

    def _show_stats(self):
        """Display session statistics."""
        stats = self.get_session_stats()
        
        if self.console:
            stats_text = f"""
[bold]Session Statistics:[/bold]
• Total Messages: [cyan]{stats.get('total_messages', 0)}[/cyan]
• Your Messages: [blue]{stats.get('user_messages', 0)}[/blue]
• AI Responses: [green]{stats.get('assistant_messages', 0)}[/green]
• System Messages: [yellow]{stats.get('system_messages', 0)}[/yellow]
• Total Characters: [magenta]{stats.get('total_characters', 0):,}[/magenta]
• Context Items: [red]{stats.get('context_items', 0)}[/red]
            """
            self.console.print(Panel(stats_text, title="Statistics", border_style="magenta"))
        else:
            print(f"\nSession Statistics:")
            print(f"• Total Messages: {stats.get('total_messages', 0)}")
            print(f"• Your Messages: {stats.get('user_messages', 0)}")
            print(f"• AI Responses: {stats.get('assistant_messages', 0)}")
            print(f"• Context Items: {stats.get('context_items', 0)}")

    def _show_search_results(self, query):
        """Display search results."""
        results = self.search_messages(query)
        
        if not results:
            if self.console:
                self.console.print(f"[yellow]No results found for: {query}[/yellow]")
            else:
                print(f"No results found for: {query}")
            return
        
        if self.console:
            table = Table(title=f"Search Results for: {query}")
            table.add_column("#", width=3)
            table.add_column("Role", width=10)
            table.add_column("Snippet", style="dim")
            
            for i, result in enumerate(results[:10], 1):
                table.add_row(str(i), result['role'], result['snippet'])
            
            self.console.print(table)
        else:
            print(f"\nSearch Results for: {query}")
            for i, result in enumerate(results[:10], 1):
                print(f"{i}. [{result['role']}] {result['snippet']}")

    # Jupyter integration methods
    
    def display_history(self, limit=None):
        """Display conversation history (Jupyter-friendly)."""
        messages_to_show = self.messages[-limit:] if limit else self.messages
        
        if self.console:
            for msg in messages_to_show:
                self._print_message(msg['role'], msg['content'])
        else:
            for msg in messages_to_show:
                print(f"\n{msg['role'].title()}: {msg['content']}")

    def to_markdown(self):
        """Export conversation to markdown format."""
        markdown_lines = [f"# Chat Session - {self.model}\n"]
        
        if self.project_name:
            markdown_lines.append(f"**Project:** {self.project_name}\n")
        
        markdown_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        for msg in self.messages:
            role_emoji = {'user': '👤', 'assistant': '🤖', 'system': '🔧'}
            emoji = role_emoji.get(msg['role'], '•')
            markdown_lines.append(f"\n## {emoji} {msg['role'].title()}\n")
            markdown_lines.append(f"{msg['content']}\n")
        
        return '\n'.join(markdown_lines)
    
    def token_counts(text):
        # Load the DeepSeek-R1 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

        # Tokenize the input text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Count the number of tokens
        token_count = len(tokens)

        return token_count

# Main application (enhanced)
def main():
    """Enhanced main application with rich interface."""
    chat_manager = OllamaChat(model='deepseek-r1:7b', visual_mode='rich')

    while True:
        try:
            chat_manager.display_session_selector()
            
            if chat_manager.console:
                choice = Prompt.ask("\n[bold]Enter your choice[/bold]", default="N")
            else:
                choice = input("\nEnter your choice (N, 1, 2, ..., Q): ").strip()
            
            choice = choice.lower()

            if choice in ['q', 'quit']:
                if chat_manager.visual_mode != 'silent':
                    chat_manager._print_success("Goodbye! 👋")
                break
            elif choice == 'n' or choice == '':
                # Start new session
                if chat_manager.console:
                    system_prompt = Prompt.ask("[yellow]System prompt[/yellow] (optional)", default="")
                    session_name = Prompt.ask("[cyan]Session name[/cyan] (optional)", default="")
                else:
                    system_prompt = input("System prompt (optional): ").strip()
                    session_name = input("Session name (optional): ").strip()
                
                chat_manager.start_new_session(
                    system_prompt=system_prompt if system_prompt else None,
                    session_name=session_name if session_name else None
                )
                chat_manager.start_interactive_chat()
            else:
                # Load existing session
                try:
                    sessions = chat_manager.list_sessions()
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < len(sessions):
                        if chat_manager.load_session(sessions[choice_index]):
                            chat_manager.start_interactive_chat()
                    else:
                        chat_manager._print_error("Invalid session number")
                except ValueError:
                    chat_manager._print_error("Invalid input. Enter N, Q, or a number.")
        
        except KeyboardInterrupt:
            if chat_manager.visual_mode != 'silent':
                chat_manager._print_success("\nGoodbye! 👋")
            break
        except Exception as e:
            chat_manager._print_error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()