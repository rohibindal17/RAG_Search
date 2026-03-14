"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._llm_with_tools = None
        self._tools_map = {}

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self):
        retriever = self.retriever

        @tool
        def retriever_tool(query: str) -> str:
            """Fetch relevant passages from the indexed document corpus. Use this first for user-provided documents."""
            docs: List[Document] = retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        @tool
        def wikipedia_tool(query: str) -> str:
            """Search Wikipedia for general knowledge not found in the document corpus."""
            return wiki.run(query)

        return [retriever_tool, wikipedia_tool]

    def _setup(self):
        """Bind tools directly to LLM — avoids LangGraph/Groq schema issues"""
        tools = self._build_tools()
        self._tools_map = {t.name: t for t in tools}
        self._llm_with_tools = self.llm.bind_tools(tools)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Manual ReAct loop: avoids create_react_agent serialization issues with Groq"""
        if self._llm_with_tools is None:
            self._setup()

        system = SystemMessage(content=(
            "You are a helpful RAG agent. "
            "Use 'retriever_tool' for user-provided documents first. "
            "Use 'wikipedia_tool' for general knowledge. "
            "Return only the final useful answer."
        ))
        messages = [system, HumanMessage(content=state.question)]

        # ReAct loop — max 5 iterations to prevent infinite loops
        for _ in range(5):
            response: AIMessage = self._llm_with_tools.invoke(messages)
            messages.append(response)

            # No tool calls → model is done
            if not response.tool_calls:
                break

            # Execute each tool call and append results
            for tc in response.tool_calls:
                tool_fn = self._tools_map.get(tc["name"])
                if tool_fn:
                    tool_result = tool_fn.invoke(tc["args"])
                else:
                    tool_result = f"Unknown tool: {tc['name']}"

                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"]
                ))

        # Last message is the final answer
        answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )