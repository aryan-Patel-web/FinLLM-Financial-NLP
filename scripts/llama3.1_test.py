import ollama 
prompt = '''
Classify the following statements as either:
premise
claim 
Return only one word 

Statement: 
Exercise improves health.
'''
response = ollama.chat(model="llama3.1:8b-instruct-q3_K_M",messages=[{"role": "user","content": prompt}])
print(response["message"]["content"])
