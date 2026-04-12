
LLM_SYSTEM_PROMPT = """You're a RAG helpful assistant. Your answers should only contain asked information"""

QUERY_ENHANCE_SPELL_PROMPTf = lambda query: f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.

User query: {query}
"""

QUERY_REWRITE_PROMPTf = lambda query: f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: {query}
"""

QUERY_EXPAND_PROMPTf = lambda query: f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: {query}
"""

RERANK_SEARCH_RESULTSf = lambda query, doc: f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""

BATCH_RERANK_SEARCH_RESULTSf = lambda query, docs: f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{docs}

Return ONLY the movie IDs (0-based) in order of relevance (best match first). Include ALL provided movie IDs in result. Return a valid JSON list, nothing else. 

For example, result for five arbitrary movies may look like this:
[4, 3, 0, 1, 2]

Ranking:"""

JUDGE_SEARCH_RESULTSf = lambda query, docs: f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{docs}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Include ALL provided movie IDs in result. Return a valid JSON list, nothing else. For example:
[2, 0, 3, 2, 0, 1]"""

RAG_ANSWER_QUESTIONf = lambda question, docs: f"""Answer the user's question based on the provided movies.

Question: {question}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

RAG_SUMMARIZE_RESULTSf = lambda query, docs: f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

RAG_ANSWER_wCITATIONSf = lambda query, docs: f"""Answer the query below and give information based on the provided documents.

If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

MERGE_QUERY_IMAGE = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

