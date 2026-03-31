import os
import base64
import fitz  # PyMuPDF
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

def extract_text_chunks(pdf_path, chunk_size=500):
    """Extract text from PDF in chunks."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1,
                    "type": "text"
                })
    return chunks

def extract_and_caption_images(pdf_path):
    """Extract images from PDF and caption them with GPT-4o Vision."""
    doc = fitz.open(pdf_path)
    image_data = []
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            img_ext = base_image["ext"]
            print(f"  Captioning image on page {page_num + 1}...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{img_ext};base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "You are analysing a technical document. Describe this image in detail, including any labels, components, measurements, or annotations visible. Be specific and technical."
                        }
                    ]
                }],
                max_tokens=300
            )
            caption = response.choices[0].message.content
            image_data.append({
                "caption": caption,
                "page": page_num + 1,
                "img_index": img_index,
                "img_b64": img_b64,
                "type": "image"
            })
    return image_data

def embed_and_store(chunks, image_data, pdf_name, collection):
    """Embed all content and store in ChromaDB."""
    all_texts = []
    all_ids = []
    all_metadata = []
    for i, chunk in enumerate(chunks):
        all_texts.append(chunk["text"])
        all_ids.append(f"{pdf_name}_text_{i}")
        all_metadata.append({"page": chunk["page"], "type": "text", "source": pdf_name})
    for i, img in enumerate(image_data):
        all_texts.append(img["caption"])
        all_ids.append(f"{pdf_name}_image_{i}")
        all_metadata.append({
            "page": img["page"],
            "type": "image",
            "source": pdf_name,
            "img_b64": img["img_b64"]
        })
    collection.add(documents=all_texts, ids=all_ids, metadatas=all_metadata)
    print(f"Stored {len(all_texts)} chunks ({len(chunks)} text, {len(image_data)} images)")

def ingest(pdf_path, custom_name=None):
    pdf_name = custom_name if custom_name else os.path.basename(pdf_path).replace(".pdf", "")
    print(f"Ingesting: {pdf_path}")
    
    # Each document gets its own collection
    doc_collection = chroma.get_or_create_collection(
        f"doc_{pdf_name}",
        embedding_function=openai_ef
    )
    # Also add to the master collection for "search all" queries
    master_collection = chroma.get_or_create_collection(
        "all_documents",
        embedding_function=openai_ef
    )
    
    print("Extracting text...")
    chunks = extract_text_chunks(pdf_path)
    print("Extracting & captioning images...")
    image_data = extract_and_caption_images(pdf_path)
    
    if not chunks and not image_data:
        print("No content found in PDF — it may be a scanned document.")
        return
    
    print("Embedding and storing...")
    embed_and_store(chunks, image_data, pdf_name, doc_collection)
    embed_and_store(chunks, image_data, pdf_name, master_collection)
    print("Done!")

if __name__ == "__main__":
    import sys
    ingest(sys.argv[1])