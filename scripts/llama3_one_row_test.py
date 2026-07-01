import pandas as pd
import ollama
df = pd.read_csv(
"data/finarg_ecc_auc_train.csv"
)
text = df.iloc[0]["text"]
prompt = f'''
You are an argument mining expert.
Classify the following statement as either: 
premise
claim
Return only one word
sentence:
{text}
'''
response = ollama.chat(model="llama3:8b", messages=[{"role": "user", "content": prompt}])
print("Text:")
print(text)
print("\nPrediction:")
print(response["message"]["content"])