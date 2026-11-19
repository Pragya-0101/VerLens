import streamlit as st
from verlens import ask

st.set_page_config(page_title="VerLens", page_icon="🔍")
st.title("VerLens")
st.caption("Version-aware docs Q&A")

lib = st.sidebar.selectbox("Library", ["sklearn"])
ver = st.sidebar.selectbox("Version", ["auto", "1.5", "1.9"])

if "h" not in st.session_state:
    st.session_state.h = []

for m in st.session_state.h:
    with st.chat_message(m["role"]):
        st.markdown(m["txt"])

q = st.chat_input("Ask about sklearn...")
if q:
    st.session_state.h.append({"role": "user", "txt": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            r = ask(q, None if ver == "auto" else ver)
        st.markdown(r["answer"])
        st.caption(f"Answered from sklearn {r['version']}")
        with st.expander("Sources"):
            for s in r["sources"]:
                st.write(s)
    st.session_state.h.append({"role": "assistant", "txt": r["answer"]})