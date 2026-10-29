from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

versions = ["1.5", "1.9"]
latest = "1.9"
other = lambda v: "1.5" if v == "1.9" else "1.9"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(embedding_function=embedding, persist_directory="verlens_vec", collection_name="verlens")
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser = StrOutputParser()

def get_context(query, ver):
    res = vector_store.similarity_search(query, filter={"version": ver}, k=5)
    return "\n\n".join(c.page_content for c in res)

ver_temp = ChatPromptTemplate.from_template(
"""Which sklearn version does the question below ask about?
The only available versions are 1.5 and 1.9.

Rules:
- Only a number used as a version counts (like "in 1.5", "v1.9", "version 1.5", "sklearn 1.9").
- A number used for anything else is NOT a version (for example "in 1.5 lines", "alpha=1.5", "1.5 times").
- If the question names no version, answer none.

Reply with ONLY one of: 1.5, 1.9, none. No other words.

Question: {question}""")

final_prompt = ChatPromptTemplate.from_template(
"""Answer the question using ONLY the context below. Never use your own knowledge.

Rules:
- Start directly with the answer. Do not say "here is the definition" or "based on the query".
- Add one or two short examples only if they help.
- If something has similar counterparts, mention them briefly as "similar".
- Use a table only if it really makes the answer clearer.
- If the context is empty or does not contain the answer, reply only: Not found in the docs

Main context (sklearn {version}):
{context}

Other version context (sklearn {other}), use it ONLY to check for differences:
{other_context}

At the end, add one line:
- If the two contexts clearly differ for this question, say what changed between {other} and {version}.
- Otherwise write: No major difference found between {other} and {version} in the docs.

Question: {question}""")

fix = RunnableLambda(lambda r: r.strip() if r.strip() in versions else latest)

chain_par = RunnableParallel({
    "version": ver_temp | model | parser | fix,
    "question": lambda x: x["question"],
})

add_cntx = RunnableLambda(lambda x: {
    "question": x["question"],
    "version": x["version"],
    "context": get_context(x["question"], x["version"]),
    "other": other(x["version"]),
    "other_context": get_context(x["question"], other(x["version"])),
})

chain = chain_par | add_cntx | final_prompt | model | parser

question = input("Enter your query ")
print(chain.invoke({"question": question}))