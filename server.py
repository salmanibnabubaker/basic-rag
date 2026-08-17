documents = [
    "PENTO WPC doors are waterproof.",
    "PENTO doors come with a 15-year warranty.",
    "PENTO doors are termite resistant."
]

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    api_key="YOUR_KEY"
)
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

docs = [
    Document(page_content=text)
    for text in documents
]

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

results = vectorstore.similarity_search(
    "What warranty is provided?",
    k=2
)

for doc in results:
    print(doc.page_content)

from openai import OpenAI

client = OpenAI(api_key="YOUR_KEY")

context = "\n".join(
    [doc.page_content for doc in results]
)

prompt = f"""
Answer only from context.

Context:
{context}

Question:
What warranty is provided?
"""

response = client.responses.create(
    model="gpt-5",
    input=prompt
)

print(response.output_text)