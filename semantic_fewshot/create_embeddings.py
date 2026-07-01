# create_embeddings.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np



model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

df = pd.read_csv(
    "data/finarg_ecc_auc_train.csv"
)

sentences = df["text"].tolist()

embeddings = model.encode(
    sentences,
    show_progress_bar=True
)

print(embeddings.shape)
np.save(
    "semantic_fewshot/mpnet_embeddings.npy",
    embeddings
)