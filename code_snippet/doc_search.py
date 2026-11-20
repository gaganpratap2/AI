import os 
from openai import OpenAI
from dotenv import load_dotenv
import numpy

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

embed_model = os.getenv("EMBED_MODEL")
docs_folder = os.getenv("DB_FOLDER")

# get cosine simalirity
def cosine_similarity(A , B):
    dot_product = numpy.dot(A , B)
    nom_len_a = numpy.linalg.norm(A)
    nom_len_b = numpy.linalg.norm(B)

    return dot_product / (nom_len_a * nom_len_b)


# add embedding to docs
def add_embedding(documents):
    for document in documents:
        vector = get_embedding(document["text"])
        document["vector"] = vector

    return documents

# get embedding
def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )

    return response.data[0].embedding


# load the documents

def load_documents(folder = docs_folder):
    documents = []

    for file in sorted(os.listdir(folder)):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(folder , file)
        with open(path , "r" , encoding="utf-8") as f:
            documents.append(
                {
                    "source":file,
                    "text":f.read()
                }
            )
    return documents

# search top k documents from user query

def search(query , documents , top_k = 3):
    query_vector = get_embedding(query)

    result = []
    for document in documents:
        score = cosine_similarity(query_vector , document["vector"])
        result.append(
            {
                "score": float(score),
                "source":document["source"],
                "text":document["text"],
                "size":len(document["text"])
            }
        )

    result.sort(key = lambda record: record["score"] , reverse=True)
    return result[:top_k]