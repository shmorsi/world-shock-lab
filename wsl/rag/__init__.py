"""Ask the decisions: retrieval-augmented answers from Fed and Bank of Canada statements.

The corpus is every policy statement behind the lab's event lists (``corpus``). Retrieval
is keyword (BM25), embeddings (a static model that also runs in the browser, and a
transformer), and their fusion (``retrieve``). The comparison was fixed in
``config/rag_protocol.yaml`` and tagged ``rag-v1`` before any retriever was written
(``evaluate``). Answers cite paragraphs and may not print a number that is not in a cited
paragraph or in the lab's results (``answer``).
"""
