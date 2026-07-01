import pandas as pd
import ollama 
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm

df = pd.read_csv("data/finarg_ecc_auc_train.csv")
df = df.sample(n=1000, random_state=42)
print(f"Test on {len(df)} rows...\n")

start_time = time.time()

texts = []
actual_labels = []
predicted_labels = []

for _, row in tqdm(df.iterrows(),total = len(df)):
    text = str(row["text"])
    true_label = str(row["answer"]).strip().lower()
    prompt = f'''
You are an argument mining expert.
Task :
Classify the following statement as either:
premise
claim

Return only one word

sentence: 
{text}
'''
    try:
        response = ollama.chat(model="gemma3n:e4b", messages=[{"role": "user", "content": prompt}])
        prediction = (response["message"]["content"].strip().lower())

        if "premise" in prediction:
            prediction = "premise"
        elif "claim" in prediction:
            prediction = "claim"
        else:
            prediction = "unknown"
    
    except Exception as e:
        print(f"Error:",e)
        prediction = "unknown"
    
    texts.append(text)
    actual_labels.append(true_label)
    predicted_labels.append(prediction)

results_df = pd.DataFrame({
    "text": texts,
    "actual": actual_labels,
    "predicted": predicted_labels
})
print("\n")
print(results_df)


valid_df = results_df[results_df["predicted"] != "unknown"]
accuracy = accuracy_score(valid_df["actual"], valid_df["predicted"])

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
print("----------------------------------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")
print(f"Micro F1 : {micro_f1:.4f}")

results_df.to_csv("results/gemma4_sample_results.csv", index=False)
print("Results saved to results/gemma4_sample_results.csv")

end_time = time.time()

print(f"Total execution time: {end_time - start_time:.2f} seconds")

