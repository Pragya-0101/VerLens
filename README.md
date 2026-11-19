# VerLens

Version-aware RAG for library docs. Ask about sklearn and VerLens answers from the **right version** (1.5 or 1.9), then tells you what changed between them.

Most docs chatbots mix versions and give answers that are wrong for the version you use. VerLens fixes that with version metadata + filtered retrieval.

![demo](demo.gif)

## Features
- **Auto version detection:** Gemini reads the question and picks the version ("in 1.5", "v1.9", "sklearn==1.9"). Numbers that aren't versions (`alpha=1.5`, `1.5 times`) are ignored.
- **Manual override:** pick the version from the dropdown instead.
- **Filtered retrieval:** only chunks tagged with that version are searched.
- **Change detection:** the answer ends with what changed between the two versions.
- **Sources shown** for every answer.

## How it works
```
question -> detect version -> Chroma search (filter: version) -> Gemini answer
                                  \-> same search on other version -> diff line
```

| file | job |
|---|---|
| `ingest.py` | loads docs, tags chunks with `version`, embeds (all-MiniLM-L6-v2) into Chroma |
| `verlens.py` | version router, retrieval, answer chain |
| `app.py` | Streamlit UI (library + version dropdowns) |
| `eval.py`, `eval_qs.py` | evaluation on 24 questions |

## Eval
| test | score |
|---|---|
| Router accuracy (includes traps like `alpha=1.5`) | 24/24 |
| Retrieval hit, top-5 (answer keyword found in retrieved chunks) | 21/24 |

Questions include vague ones where the keyword isn't in the question. Run with `python eval.py`.

## Run it
1. `pip install -r requirements.txt`
2. Put the sklearn docs in `lib/sklearn/1.5` and `lib/sklearn/1.9`
3. Create `.env` with `GOOGLE_API_KEY=your_key`
4. `python ingest.py` (run once)
5. `streamlit run app.py`

## Stack
LangChain, ChromaDB, HuggingFace embeddings, Gemini, Streamlit

## Roadmap
- Add pandas and more versions
- Tune k / chunk size for retrieval