"""
Run this ONCE to build the vector store, before running graph.py.

    python build_vectorstore.py

Needs OPENAI_API_KEY in a .env file in this folder (used to create the embeddings).
This only needs to be re-run if clinical_guidelines.pdf changes.
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFLoader("clinical_guidelines.pdf")
pages = loader.load()
full_text = "\n".join(page.page_content for page in pages)

# PDF extraction turns every visual line wrap into a newline, with no way
# to tell "mid-paragraph wrap" apart from "real section break" just by
# counting newlines. So instead of cutting by character count, detect the
# actual heading lines directly, and group everything between two headings
# as that heading's body.
KNOWN_HEADINGS = [
    "DIABETES MANAGEMENT GUIDELINES",
    "HEART FAILURE MANAGEMENT GUIDELINES",
    "CHRONIC KIDNEY DISEASE MANAGEMENT GUIDELINES",
    "HYPERTENSION MANAGEMENT GUIDELINES",
    "ASTHMA MANAGEMENT GUIDELINES",
]

lines = full_text.split("\n")
sections = {}
current_heading = None
for line in lines:
    line = line.strip()
    if line in KNOWN_HEADINGS:
        current_heading = line
        sections[current_heading] = []
    elif current_heading:
        sections[current_heading].append(line)

# One chunk per topic. Clean, complete, no character-count cutting,
# no overlap needed, and the line wraps inside each body are collapsed
# into single spaces so no stray \n characters end up in the stored text.
chunks = []
for heading, body_lines in sections.items():
    body = " ".join(body_lines)
    chunks.append(Document(page_content=f"{heading}\n{body}"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./guideline_chroma_db",
)

print(f"Built vector store with {len(chunks)} chunks (one per guideline topic), saved to ./guideline_chroma_db")
