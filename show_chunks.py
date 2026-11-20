from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

vectorstore = Chroma(
    persist_directory="./guideline_chroma_db",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

data = vectorstore.get(include=["documents", "embeddings"])

for i, text in enumerate(data["documents"]):
    print(f"--- Chunk {i + 1} ---")
    print(text)
    print(f"(stored as {len(data['embeddings'][i])} numbers)")
    print()