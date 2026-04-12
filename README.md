A bunch of CLIs for searching relevant movies from json database.

- keyword_search_cli - bm25 tf-idf keyword search

- semantic_search_cli - semantic search by using text embeddings, cosine similarity and text chunking

- hybrid_search_cli - combining keyword and semantic scores into one, by using min-max normalization and weights / by using rrf (rank based) algorithm; applying llm to enhance user query, rerank search results

- evaluation_cli - evaluating rrf search quality with precision, recall, f1 metrics based on "golden" dataset

- augmented_generation_cli - utilizing basic rag techniques for searching: summarization, question answering, making citations

- describe_image_cli - test script for passing image into LLM and rewriting query based on it

- multimodal_search_cli - utilizing CLIP text-image embedding model for searching with image
