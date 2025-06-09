from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
import os
import sys
from langchain import hub

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("No OPENAI_API_KEY")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

model = init_chat_model(model="gpt-4.1", model_provider="openai")

template = """
Respond only with git commands and nothing else to solve the user's problem. One command per line.

Context:
{context}

Question:
{question}
"""

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}

prompt = PromptTemplate.from_template(template)

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": input("Enter a git query: ")})

print(f'{result["answer"]}')

# prompt = prompt_template.invoke({"text": "I need to revert the file main.py to three commits ago."})

# response = model.invoke(prompt)
# print(response.content)