# examples.py
# Demonstrates key use cases of the OllamaChat library when pulled into a notebook or script.

from ollama_chat import OllamaChat


def example_new_session():
    """
    Start a new session with a system prompt and send a question.
    """
    chat = OllamaChat(
        model='deepseek-coder:7b',
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.start_new_session(
        system_prompt="You are a data-science mentor who explains things first-principles style.",
        session_name="binary-search-deep-dive"
    )
    response = chat.send("Walk me through how binary search works.")
    print("AI:", response)


def example_load_and_resume():
    """
    Load and resume an existing session.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    if chat.load_session("binary-search-deep-dive.json"):
        reply = chat.send("What’s the worst-case time complexity again?")
        print("AI:", reply)


def example_inject_context():
    """
    Add text or file context to the session.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.load_session("binary-search-deep-dive.json")
    # Add inline context
    chat.add_context(
        "Quick-Sort Note",
        "Quick-sort on average runs in O(n log n) time by picking a pivot and partitioning..."
    )
    # Or add from a file
    # chat.add_file_context("research/algorithm_notes.txt", title="Algorithm Notes")
    response = chat.send("Compare binary search and quick-sort given the new context.")
    print("AI:", response)


def example_search_and_extract():
    """
    Search past messages and extract code blocks from the last response.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.load_session("binary-search-deep-dive.json")
    results = chat.search_messages("binary search")
    print("Search Results:", results)

    code_blocks = chat.extract_code_blocks()
    print("Code Blocks:", code_blocks)


def example_stats_and_history():
    """
    Retrieve and print session statistics and recent history.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.load_session("binary-search-deep-dive.json")
    stats = chat.get_session_stats()
    print("Session Stats:", stats)
    # Display last 5 messages in console
    chat.display_history(limit=5)


def example_export_markdown():
    """
    Export the full conversation to a Markdown file.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.load_session("binary-search-deep-dive.json")
    markdown = chat.to_markdown()
    output_path = "binary_search_session.md"
    with open(output_path, "w") as f:
        f.write(markdown)
    print(f"Exported conversation to {output_path}")


def example_silent_batch():
    """
    Run batch queries in silent mode (no live output).
    """
    chat = OllamaChat(
        project_name='batch-runs',
        visual_mode='silent'
    )
    chat.start_new_session(session_name="batch-queries")
    queries = ["Explain PCA", "Give me pseudocode for k-means"]
    answers = [chat.send(q) for q in queries]
    print("Batch answers:", answers)


def example_interactive_loop():
    """
    Simple REPL loop driven by input(); type 'quit' to exit.
    """
    chat = OllamaChat(
        project_name='notebook-demo',
        visual_mode='plain'
    )
    chat.start_new_session(session_name="interactive-loop")
    print("Starting interactive loop. Type 'quit' to exit.")
    while True:
        user_q = input("You ▶ ")
        if user_q.lower() in ('quit', 'exit'):
            break
        ai_response = chat.send(user_q)
        print(f"AI ▶ {ai_response}\n")


if __name__ == "__main__":
    example_new_session()
    example_load_and_resume()
    example_inject_context()
    example_search_and_extract()
    example_stats_and_history()
    example_export_markdown()
    example_silent_batch()
    # Uncomment to run the interactive loop example:
    # example_interactive_loop()
