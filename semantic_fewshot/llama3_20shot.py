import pandas as pd
import numpy as np
import ollama
import time 

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# LOAD TRAIN / TEST SPLITS

train_df = pd.read_csv(
    "data/train_split.csv"
)

test_df = pd.read_csv(
    "data/test_split.csv"
)

start_time = time.time()

# # For debugging
# test_df = test_df.sample(
#     n=100,
#     random_state=42
# )

# LOAD TRAIN EMBEDDINGS


embeddings = np.load(
    "semantic_fewshot/train_embeddings.npy"
)

st_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


# RETRIEVAL


def retrieve_examples(query, k=20):

    query_embedding = st_model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-k:][::-1]

    return train_df.iloc[top_indices]


# PROMPT


def build_prompt(query, examples):

    prompt = """
You are an expert in financial argument mining.

Task:
Classify the given sentence into exactly one category.

premise = supporting evidence, reason, fact, justification

claim = conclusion, assertion, opinion, recommendation

Return ONLY one word:
premise
or
claim

Examples:

"""

    for _, row in examples.iterrows():

        prompt += f"""
Sentence: {row['text']}
Label: {row['answer']}
"""

    prompt += f"""

Now classify this sentence:

Sentence: {query}

Label:
"""

    return prompt


# INFERENCE

actual = []
predicted = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

    sentence = row["text"]

    examples = retrieve_examples(
        sentence,
        k=20
    )

    prompt = build_prompt(
        sentence,
        examples
    )

    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    prediction = response["message"]["content"].lower()

    if "premise" in prediction:
        prediction = "premise"

    elif "claim" in prediction:
        prediction = "claim"

    else:
        prediction = "unknown"

    actual.append(
        str(row["answer"]).strip().lower()
    )

    predicted.append(prediction)

# RESULTS

results_df = pd.DataFrame({
    "actual": actual,
    "predicted": predicted
})

valid_df = results_df[
    results_df["predicted"] != "unknown"
]

accuracy = accuracy_score(
    valid_df["actual"],
    valid_df["predicted"]
)

macro_f1 = f1_score(
    valid_df["actual"],
    valid_df["predicted"],
    average="macro"
)

micro_f1 = f1_score(
    valid_df["actual"],
    valid_df["predicted"],
    average="micro"
)

print("\nRESULTS")
print("-----------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Micro F1 : {micro_f1:.4f}")
results_df.to_csv(
    "results/llama3_20shot_results.csv",
    index=False
)

end_time = time.time()

print(f"Total execution time: {end_time - start_time:.2f} seconds")

