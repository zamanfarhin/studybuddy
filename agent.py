"""
Agent module: Agentic decision layer for routing queries.
"""

from openai import OpenAI
from rag import RAGEngine
import json


class StudyBuddyAgent:
    def __init__(self, openai_api_key: str, rag_engine: RAGEngine):
        self.client = OpenAI(api_key=openai_api_key)
        self.rag_engine = rag_engine
        self.conversation_history = []

    def decide_action(self, user_query: str) -> dict:
        """
        Decide whether to:
        (a) answer from memory/general knowledge
        (b) retrieve from documents
        (c) hybrid - use both

        Uses function calling to guarantee structured output.
        """
        has_documents = self.rag_engine.get_document_count() > 0

        # Define the function for structured output
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_query",
                    "description": "Classify the user's query to determine the best way to answer it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "2-3 sentences explaining why this classification was chosen"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["memory", "retrieve", "hybrid"],
                                "description": "The classification: memory for general knowledge, retrieve for document-specific, hybrid for both"
                            }
                        },
                        "required": ["reasoning", "action"]
                    }
                }
            }
        ]

        classification_prompt = f"""You are a query classifier. Analyze this question and decide HOW to answer it.

USER QUESTION: "{user_query}"

DOCUMENTS UPLOADED: {has_documents}

CLASSIFICATION RULES (follow these strictly):

1. **memory** - Use for GENERAL KNOWLEDGE questions:
   - Definitions, concepts, explanations (e.g., "What is machine learning?", "Explain photosynthesis")
   - Common facts anyone educated would know
   - Questions with NO reference to uploaded documents
   - If no documents are uploaded, ALWAYS use memory

2. **retrieve** - Use ONLY when the question EXPLICITLY references uploaded documents:
   - "What does my document say about..."
   - "According to the PDF..."
   - "In chapter 3..."
   - "Based on what I uploaded..."
   - Questions that can ONLY be answered by the specific uploaded content

3. **hybrid** - Use when BOTH are needed:
   - "How does [concept from my doc] compare to [general knowledge]?"
   - "Explain [topic from doc] in broader context"
   - Questions needing document facts PLUS general explanation

IMPORTANT: Default to "memory" for general questions. Only use "retrieve" when the user explicitly asks about their documents.

Call the classify_query function with your reasoning and decision."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": classification_prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "classify_query"}},
            temperature=0
        )

        # Extract the function call result
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)

        action = result.get("action", "memory")
        reasoning = result.get("reasoning", "No reasoning provided")

        # Validate action is one of the allowed values
        if action not in ["memory", "retrieve", "hybrid"]:
            action = "memory"
            reasoning = f"Invalid action '{action}' returned, defaulting to memory. Original reasoning: {reasoning}"

        # Handle case where retrieval is requested but no documents exist
        if not has_documents and action in ["retrieve", "hybrid"]:
            action = "memory"
            reasoning += " (Adjusted: No documents uploaded, using general knowledge instead)"

        return {
            "action": action,
            "reasoning": reasoning
        }

    def generate_response(self, user_query: str, decision: dict) -> str:
        """Generate a response based on the decision."""
        action = decision["action"]
        context = ""

        # Retrieve if needed
        if action in ["retrieve", "hybrid"]:
            retrieved_docs = self.rag_engine.retrieve(user_query, n_results=3)
            if retrieved_docs:
                context = "Relevant information from uploaded documents:\n\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    context += f"[Source: {doc['source']}]\n{doc['content']}\n\n"

        # Build the prompt
        if action == "memory":
            system_prompt = """You are StudyBuddy, a helpful study assistant.
Answer the user's question using your general knowledge.
Be clear, educational, and helpful."""
        elif action == "retrieve":
            system_prompt = f"""You are StudyBuddy, a helpful study assistant.
Answer the user's question based on the following context from their uploaded documents.
If the context doesn't contain relevant information, say so honestly.

{context}"""
        else:  # hybrid
            system_prompt = f"""You are StudyBuddy, a helpful study assistant.
Answer the user's question using both your general knowledge AND the following context from their uploaded documents.
Integrate both sources naturally in your response.

{context}"""

        # Add conversation history for context
        messages = [{"role": "system", "content": system_prompt}]

        # Add recent conversation history (last 6 exchanges)
        for msg in self.conversation_history[-12:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_query})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def chat(self, user_query: str) -> tuple[str, dict]:
        """
        Main chat method that decides action and generates response.
        Returns (response, decision)
        """
        decision = self.decide_action(user_query)
        response = self.generate_response(user_query, decision)
        return response, decision

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
