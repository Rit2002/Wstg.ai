import streamlit as st
from rag_pipeline import generate_answer_stream

st.set_page_config(
    page_title="OWASP WSTG Assistant",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ OWASP WSTG Assistant")
st.caption(
    "Ask questions about web application security testing "
    "using the OWASP Web Security Testing Guide."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(
                        f'**[{source["id"]}]** '
                        f'{source["source"]} | '
                        f'{source["heading"]}'
                    )

query = st.chat_input("Ask a WSTG security testing question...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer_area = st.empty()
        full_answer = ""
        metadata = None

        try:
            with st.spinner("Retrieving relevant WSTG guidance..."):
                events = generate_answer_stream(query)

                for event in events:
                    if event["type"] == "token":
                        full_answer += event["content"]
                        answer_area.markdown(full_answer + " ▌")

                    elif event["type"] == "done":
                        metadata = event
            
            if metadata:
                st.caption(
                    "Citation validation: "
                    + metadata["citation_validation"]["status"]
                )

                with st.expander("Retrieved sources"):
                    for source in metadata["sources"]:
                        st.markdown(
                            f'**[{source["id"]}]** '
                            f'{source["source"]} | '
                            f'{source["heading"]}'
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": metadata["sources"] if metadata else [],
            })

        except Exception as exc:
            st.error(f"Generation failed: {exc}")