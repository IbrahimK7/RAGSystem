import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "top_100_highest_grossing_films.pdf"
DB_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
llm = ChatOllama(model="llama3.1:8b", temperature=0)

SYSTEM_PROMPT = (
    "You answer questions about the top 100 highest-grossing films of all time worldwide. "
    "Use ONLY the context provided. Do not use outside knowledge and do not guess. "
    "If the context does not contain the answer, say so clearly. "
    "End your answer with a new line in the form: Source: page N"
)


def build_db() -> Chroma:
    # Reuse the saved database if it exists; delete the chroma_db folder to rebuild
    if DB_DIR.exists():
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings, collection_name="films")

    pages = PyMuPDFLoader(str(PDF_PATH)).load()
    for p in pages:  # drop the footer repeated on every page
        p.page_content = re.sub(r"^Top 100 Highest-Grossing Films - Box Office Mojo.*\nPage \d+\n", "", p.page_content)

    chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_documents(pages)
    print(f"Embedding {len(chunks)} chunks (one-time)...")
    return Chroma.from_documents(
        chunks, embeddings,
        persist_directory=str(DB_DIR), collection_name="films",
        collection_metadata={"hnsw:space": "cosine"},
    )


retriever = build_db().as_retriever(search_kwargs={"k": 4})


def ask(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[page {d.metadata['page'] + 1}]\n{d.page_content}" for d in docs)
    reply = llm.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    return reply.content


if __name__ == "__main__":
    print("Ask about the top 100 highest-grossing films. Type 'exit' to quit.")
    while (q := input("\nYour question: ").strip()).lower() not in {"exit", "quit", "q"}:
        if q:
            print("-" * 50 + "\n" + ask(q))