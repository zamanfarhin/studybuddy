"""
StudyBuddy - A Streamlit RAG application with agentic decision-making.
"""

import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd

from rag import RAGEngine
from agent import StudyBuddyAgent
from experiments import ExperimentRunner, RESULTS_FILE, TOTAL_RUNS

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="StudyBuddy",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "experiment_pdf_bytes" not in st.session_state:
    st.session_state.experiment_pdf_bytes = None

if "experiment_pdf_name" not in st.session_state:
    st.session_state.experiment_pdf_name = None

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None

if "rag_engine" not in st.session_state:
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_key_here":
        st.session_state.rag_engine = RAGEngine(openai_key)
    else:
        st.session_state.rag_engine = None

if "agent" not in st.session_state:
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_key_here" and st.session_state.rag_engine:
        st.session_state.agent = StudyBuddyAgent(openai_key, st.session_state.rag_engine)
    else:
        st.session_state.agent = None

# Sidebar
with st.sidebar:
    st.title("📚 StudyBuddy")
    st.markdown("---")

    # API Key status
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_key_here":
        st.error("⚠️ Please set your OPENAI_API_KEY in the .env file")
    else:
        st.success("✅ OpenAI API key loaded")

    st.markdown("---")

    # PDF Upload
    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload study materials",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.session_state.rag_engine:
            for uploaded_file in uploaded_files:
                # Check if file was already processed
                file_key = f"processed_{uploaded_file.name}"
                if file_key not in st.session_state:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        pdf_bytes = uploaded_file.read()
                        num_chunks = st.session_state.rag_engine.process_and_store_pdf(
                            pdf_bytes, uploaded_file.name
                        )
                        st.session_state[file_key] = True
                        # Store for experiment use
                        st.session_state.experiment_pdf_bytes = pdf_bytes
                        st.session_state.experiment_pdf_name = uploaded_file.name
                        st.success(f"✅ {uploaded_file.name}: {num_chunks} chunks indexed")
        else:
            st.warning("Cannot process PDFs without OpenAI API key")

    st.markdown("---")

    # Document stats
    if st.session_state.rag_engine:
        doc_count = st.session_state.rag_engine.get_document_count()
        st.metric("Indexed Chunks", doc_count)

    st.markdown("---")

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()

    with col2:
        if st.button("Clear Docs"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_collection()
                # Clear processed file markers
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith("processed_")]
                for k in keys_to_delete:
                    del st.session_state[k]
                st.rerun()

    st.markdown("---")

    # Experiments section
    st.subheader("🧪 Experiments")
    openai_key_exp = os.getenv("OPENAI_API_KEY")
    if st.button("Run Experiments"):
        if not st.session_state.experiment_pdf_bytes:
            st.warning("Upload a PDF first before running experiments.")
        elif not openai_key_exp or openai_key_exp == "your_openai_key_here":
            st.error("OpenAI API key required.")
        else:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(fraction: float):
                progress_bar.progress(fraction)
                done = int(fraction * TOTAL_RUNS)
                status_text.text(f"Running experiments... {done}/{TOTAL_RUNS}")

            status_text.text("Indexing PDFs for each chunk size (this may take a minute)...")
            runner = ExperimentRunner(
                openai_key_exp,
                st.session_state.experiment_pdf_bytes,
                st.session_state.experiment_pdf_name
            )
            with st.spinner("Running all experiment configurations..."):
                results = runner.run_all(progress_callback=update_progress)

            st.session_state.experiment_results = results
            progress_bar.progress(1.0)
            status_text.text(f"Done! {TOTAL_RUNS}/{TOTAL_RUNS}")
            st.success(f"Saved {len(results)} runs to {RESULTS_FILE}")

# Main content
st.title("📚 StudyBuddy")
st.caption("Your AI-powered study assistant with document understanding")

tab_chat, tab_experiments = st.tabs(["Chat", "Experiments"])

with tab_experiments:
    st.subheader("Experiment Results")
    if st.session_state.experiment_results:
        df = pd.DataFrame(st.session_state.experiment_results)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Results also saved to `{RESULTS_FILE}`")
    else:
        st.info("No experiment results yet. Click **Run Experiments** in the sidebar to start.")

with tab_chat:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "decision" in message:
                decision = message["decision"]
                action_emoji = {"memory": "🧠", "retrieve": "📄", "hybrid": "🔀"}
                action_label = {"memory": "General Knowledge", "retrieve": "Document Retrieval", "hybrid": "Hybrid"}
                st.caption(
                    f"{action_emoji.get(decision['action'], '❓')} **Decision:** {action_label.get(decision['action'], decision['action'])}"
                )

# Chat input (outside tabs so it stays pinned at bottom)
if prompt := st.chat_input("Ask me anything about your study materials..."):
    if not st.session_state.agent:
        st.error("Please configure your OpenAI API key in the .env file")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, decision = st.session_state.agent.chat(prompt)

            st.markdown(response)

            # Show decision
            action_emoji = {"memory": "🧠", "retrieve": "📄", "hybrid": "🔀"}
            action_label = {"memory": "General Knowledge", "retrieve": "Document Retrieval", "hybrid": "Hybrid"}
            st.caption(
                f"{action_emoji.get(decision['action'], '❓')} **Decision:** {action_label.get(decision['action'], decision['action'])}"
            )

        # Store assistant message with decision
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "decision": decision
        })
