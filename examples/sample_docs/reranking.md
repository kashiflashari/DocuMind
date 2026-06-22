# Re-ranking

After hybrid retrieval gathers a candidate pool, a re-ranker reorders those
candidates by relevance to the query. DocuMind uses a Cohere cross encoder, which
jointly encodes the query and each passage to produce a precise relevance score —
far more accurate than the first-stage retrieval scores, at the cost of running
one model pass per candidate. Re-ranking is what lifts the right passages into the
small top-k window that is finally shown to the language model.
