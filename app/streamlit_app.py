import sys
from pathlib import Path

# Add the project root to sys.path so 'app' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import streamlit as st

from app.config import get_settings
from app.pipeline import RAGPipeline

st.set_page_config(page_title="Cost-Efficient RAG", page_icon="🤖", layout="wide")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(get_settings())

pipeline = load_pipeline()

with st.sidebar:
    st.title("⚙️ System Stats")
    stats = pipeline.stats()
    
    st.metric("Chunks in DB", stats.get("count", 0))
    st.caption(f"**Model:** `{stats.get('embedding_model')}`")
    st.caption(f"**LLM:** `{stats.get('llm')}`")
    
    st.divider()
    
    st.title("📂 Batch Upload")
    st.markdown("Upload a directory of files. Supported: PDF, HTML, MD.")
    
    # REMOVED png, jpg, jpeg from type list
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "md", "html"], 
        accept_multiple_files=True 
    )
    
    if uploaded_files:
        if st.button(f"Ingest {len(uploaded_files)} Files", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{len(uploaded_files)})...")
                
                # Create a temp dir, but save the file with its ORIGINAL name
                import shutil
                temp_dir = tempfile.mkdtemp()
                tmp_path = Path(temp_dir) / file.name
                with open(tmp_path, "wb") as f:
                    f.write(file.getvalue())
                
                try:
                    result = pipeline.ingest_file(tmp_path)
                    success_count += 1
                except Exception as e:
                    st.error(f"Failed on {file.name}: {e}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text(f"✅ Done! Successfully ingested {success_count} files.")
            st.rerun()

st.title("🤖 RAG Document Assistant")
st.markdown("Ask questions about the ingested documents. The system will answer with citations or abstain if the answer is not in the context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Sources"):
                for cite in msg["citations"]:
                    st.markdown(f"**{cite['filename']}** (Chunk {cite['chunk_index']}) - Score: {cite['score']}")
                    st.code(cite['snippet'], language="text")

if prompt := st.chat_input("e.g., What is the SLA on the Enterprise plan?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = pipeline.ask(prompt, top_k=5)
                
                answer = res["answer"]
                citations = res["citations"]
                
                st.markdown(answer)
                
                if citations:
                    with st.expander("📚 Sources"):
                        for cite in citations:
                            st.markdown(f"**{cite['filename']}** (Chunk {cite['chunk_index']}) - Score: {cite['score']}")
                            st.code(cite['snippet'], language="text")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")