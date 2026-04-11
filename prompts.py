
LLM_ENHANCE_SYSTEM_PROMPT = """You're RAG helpful assistant. Your answers should only contain asked result without additional context and explanations"""

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