import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/finarg_ecc_auc_train.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["answer"]
)

train_df.to_csv(
    "data/train_split.csv",
    index=False
)

test_df.to_csv(
    "data/test_split.csv",
    index=False
)

model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

train_embeddings = model.encode(
    train_df["text"].tolist(),
    show_progress_bar=True
)

np.save(
    "semantic_fewshot/train_embeddings.npy",
    train_embeddings
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Embeddings shape:", train_embeddings.shape)