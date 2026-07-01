import ollama 
prompt = '''
Classify the following statements as either:
premise
claim 
Return only one word 

Statement: 
Exercise improves health.
'''
response = ollama.chat(model="gemma3n:e4b",messages=[{"role": "user","content": prompt}])
print(response["message"]["content"])
