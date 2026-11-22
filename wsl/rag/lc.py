"""The statement search as a LangChain retriever, so it drops into any LangChain chain:

    from wsl.rag.lc import StatementRetriever
    docs = StatementRetriever(kind="hybrid", k=5).invoke("When did the Fed begin tapering?")

Each Document is a statement's best-matching paragraph, with the bank, date, title and
official URL in its metadata.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from wsl.rag.retrieve import KINDS, Searcher


class StatementRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "hybrid"
    k: int = 5
    searcher: Any = Field(default=None, exclude=True)

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.searcher is None:
            self.searcher = Searcher()
        s: Searcher = self.searcher
        by_id = {st.doc_id: st for st in s.statements}
        out = []
        for c in s.passages(self.kind, query, self.k):
            st = by_id[c.doc_id]
            out.append(Document(page_content=c.body, id=c.chunk_id, metadata={
                "doc_id": c.doc_id, "bank": c.bank, "date": c.date, "title": st.title,
                "url": st.url, "header": c.header}))
        return out
