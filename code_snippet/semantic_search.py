from openai import OpenAi
from dotenv import load_dotenv
import os
import numpy

load_dotenv()

client = OpenAi(
    base_url = os.getenv("BASR_URL"),
    api_key = os.getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=text
    )

    return response.data[0].embedding

def cosine_similarity(A , B):
    dot_product = numpy.dot(A , B)
    nom_len_a = numpy.linalg.norm(A)
    nom_len_b = numpy.linalg.norm(B)

    return dot_product / (nom_len_a * nom_len_b)