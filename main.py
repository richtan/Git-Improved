import os
import sys
import subprocess
import shlex
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("No OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

model = init_chat_model(model="gpt-4.1", model_provider="openai", temperature=0)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = model.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are a Git command generator. Based on the user's natural language query, generate the appropriate Git commands."
        "\n\nRules:"
        "\n- Return only the git commands, one per line"
        "\n- Include all necessary flags and options"
        "\n- The user's operating system is MacOS Sequoia"
        "\n- Be precise and accurate with command syntax"
        "\n- If multiple commands are needed, list them in the correct order"
        "\n\nGit Documentation Context:"
        f"\n{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def parse_git_commands(response_text: str) -> List[str]:
    """Parse Git commands from the AI response."""
    lines = response_text.strip().split('\n')
    commands = []
    for line in lines:
        line = line.strip()
        if line and (line.startswith('git ') or line.startswith('cd ')):
            commands.append(line)
    return commands

def execute_command(command: str) -> Tuple[bool, str]:
    """Execute a shell command and return success status and output."""
    try:
        result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr or e.stdout}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_user_confirmation(commands: List[str]) -> bool:
    """Ask user to confirm command execution."""
    print("\n🔍 Generated Git commands:")
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    
    while True:
        response = input("\n❓ Execute these commands? (y/n/e to edit): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response in ['e', 'edit']:
            return False
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'e' (edit)")

def main():
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    
    print("🚀 Git Command Assistant")
    print("Type your Git-related queries in natural language.")
    print("Type 'quit' or 'exit' to end the session.\n")
    
    while True:
        try:
            try:
                user_input = input("💬 Git query: ").strip()
            except EOFError:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🔍 Analyzing your request...")
            
            # Generate commands using the graph
            commands_generated = False
            for step in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                stream_mode="values",
                config=config,
            ):
                if step["messages"][-1].type == "ai" and not step["messages"][-1].tool_calls:
                    response_text = step["messages"][-1].content
                    commands = parse_git_commands(response_text)
                    
                    if not commands:
                        print("❌ No valid Git commands found in response.")
                        print(f"Response: {response_text}")
                        continue
                    
                    commands_generated = True
                    
                    # Ask for user confirmation
                    if get_user_confirmation(commands):
                        print("\n⚡ Executing commands...")
                        
                        for i, command in enumerate(commands, 1):
                            print(f"\n📋 Running command {i}: {command}")
                            success, output = execute_command(command)
                            
                            if output.strip():
                                print(f"📤 Output: {output.strip()}")
                            
                            if not success:
                                print(f"❌ Command failed: {command}")
                                print(f"Error: {output}")
                                break
                            else:
                                print(f"✅ Command completed successfully")
                        
                        print("\n✨ All commands executed!")
                    else:
                        print("❌ Commands not executed.")
                    
                    break
            
            if not commands_generated:
                print("❌ Could not generate Git commands for your query.")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()