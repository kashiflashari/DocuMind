# Hybrid Search

Hybrid search combines dense vector similarity with sparse keyword (lexical) matching.
Dense retrieval captures semantic meaning even when wording differs, while keyword
search excels at exact terms, names, and rare tokens. DocuMind fuses the two rankings
with Reciprocal Rank Fusion (RRF), which sums the reciprocal of each document's rank
across retrievers and therefore needs no score calibration between them.
