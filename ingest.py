import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

root = r"C:\Users\pragy\OneDrive\Desktop\RAG\VerLens\lib\sklearn"  # <- change: folder containing 1.5 and 1.9
vers = ["1.5", "1.9"]

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vs = Chroma(embedding_function=emb, persist_directory="verlens_vec", collection_name="verlens")
vs.delete_collection()  # wipe, so re-running doesn't duplicate
vs = Chroma(embedding_function=emb, persist_directory="verlens_vec", collection_name="verlens")

split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

for v in vers:
    d = os.path.join(root, v)
    docs = []
    for dp, _, fs in os.walk(d):
        for f in fs:
            if f.endswith((".rst", ".txt", ".md")):
                p = os.path.join(dp, f)
                t = open(p, encoding="utf-8", errors="ignore").read()
                if t.strip():
                    docs.append(Document(page_content=t, metadata={
                        "library": "sklearn",
                        "version": v,
                        "source": os.path.relpath(p, d),
                    }))
    chunks = split.split_documents(docs)
    print(v, "files:", len(docs), "chunks:", len(chunks))
    for i in range(0, len(chunks), 500):
        vs.add_documents(chunks[i:i + 500])

print("total in store:", vs._collection.count())