def generate_output(
        query,
        retriever,
        llm):

    docs = retriever.retrieve(
        query
    )

    context = "\n".join(docs)

    prompt = f"""
You are a medical assistant.

Answer only from the provided context.

If information is unavailable say:

"I don't have enough information in my knowledge base."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content