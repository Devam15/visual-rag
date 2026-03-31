import os
import base64
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()
client = OpenAI()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
chroma = chromadb.PersistentClient(path="./chroma_db")

def query(question,doc_name= None, n_results=3):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    if doc_name:
        collection = chroma.get_or_create_collection(
            f"doc_{doc_name}",
            embedding_function=openai_ef
        )
        print(f"Searching in: {doc_name}")
    else:
        collection = chroma.get_or_create_collection(
            "all_documents",
            embedding_function=openai_ef
        )
        print("Searching across all documents")

    # Find most relevant chunks
    results = collection.query(query_texts=[question], n_results=n_results)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context for GPT-4o
    context = ""
    images_found = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        context += f"\n[Source {i+1} - Page {meta['page']} - Type: {meta['type']}]\n{doc}\n"
        if meta["type"] == "image":
            images_found.append(meta)

    # Build message to GPT-4o
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant answering questions about technical documents. Always cite which page your answer comes from."
    }]

    user_content = [{"type": "text", "text": f"Using the following context, answer this question: {question}\n\nContext:\n{context}"}]

    # Attach relevant images
    for img_meta in images_found:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_meta['img_b64']}"
            }
        })

    messages.append({"role": "user", "content": user_content})

    # Get answer from GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    print(f"\nAnswer:\n{answer}")

    if images_found:
        print(f"\n[Referenced {len(images_found)} image(s) from pages: {[m['page'] for m in images_found]}]")

    return answer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        print("Visual RAG ready! Type your question or 'quit' to exit.")
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() == "quit":
                break
            if question:
                query(question)