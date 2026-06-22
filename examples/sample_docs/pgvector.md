# pgvector

pgvector is a PostgreSQL extension that adds a `vector` column type and indexes for
similarity search. DocuMind stores each chunk's embedding in a `vector` column and
queries it with the cosine-distance operator `<=>`. Because the vectors live in
Postgres alongside a `tsvector` full-text column, a single database backs both the
dense and lexical halves of hybrid search, with ivfflat and GIN indexes for speed.
