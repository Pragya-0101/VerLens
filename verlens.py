from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

versions = ["1.5", "1.9"]
latest = "1.9"
other = lambda v: "1.5" if v == "1.9" else "1.9"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding, persist_directory="verlens_vec", collection_name="verlens")
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser = StrOutputParser()

ver_temp = ChatPromptTemplate.from_template(
"""Check which sklearn version this question is about. Only 1.5 and 1.9 exist.

A number only counts as a version if it's used like one ("in 1.5", "v1.9", "version 1.5", "sklearn 1.9").
Things like "alpha=1.5", "1.5 times" or "in 1.5 lines" are not versions.
If no version is mentioned, say none.

Reply with just one word: 1.5, 1.9 or none.

Question: {question}""")




final_prompt = ChatPromptTemplate.from_template(
"""Answer the question using only the context below. Don't use anything you already know.

Just start with the answer, no intro like "here is the definition" or "based on the context".
Give a small example or two if it actually helps. If there are similar things, mention them quickly as "similar".
Only use a table if it makes things clearer.
If the context is empty or doesn't have the answer, just reply: Not found in the docs

Context (sklearn {version}):
{context}

Context from sklearn {other}, only use this to spot differences:
{other_context}

Last line of the answer:
- if the two contexts clearly differ on this topic, say what changed from {other} to {version}
- if not, write: No major difference found between {other} and {version} in the docs.

Question: {question}""")

fix = RunnableLambda(lambda r: r.strip() if r.strip() in versions else latest)
detect = ver_temp | model | parser | fix
answer = final_prompt | model | parser

def get_docs(query, ver, k=5):
    return vector_store.similarity_search(query, filter={"version": ver}, k=k)

def get_context(query, ver):
    return "\n\n".join(c.page_content for c in get_docs(query, ver))

def ask(q, ver=None):
    v = ver or detect.invoke({"question": q})
    o = other(v)
    docs = get_docs(q, v)
    out = answer.invoke({
        "question": q,
        "version": v,
        "context": "\n\n".join(d.page_content for d in docs),
        "other": o,
        "other_context": get_context(q, o),
    })
    src = sorted({d.metadata.get("source", "?") for d in docs})
    return {"answer": out, "version": v, "sources": src}

if __name__ == "__main__":
    r = ask(input("Enter your query "))
    print(r["answer"], "\n", r["version"], r["sources"])