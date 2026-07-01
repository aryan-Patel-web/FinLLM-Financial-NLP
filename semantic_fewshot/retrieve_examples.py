import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/finarg_ecc_auc_train.csv")

# Load saved embeddings
embeddings = np.load(
    "semantic_fewshot/mpnet_embeddings.npy"
)

# Load MPNet model
model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


def retrieve_examples(query, k=5):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]

    return df.iloc[top_indices]


query = "The company's revenue increased significantly."

results = retrieve_examples(
    query,
    k=5
)

print(results[["text", "answer"]])