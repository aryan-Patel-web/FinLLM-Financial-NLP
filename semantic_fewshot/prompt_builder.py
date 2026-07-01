def build_prompt(query, examples):

    prompt = """
You are a financial argument classification expert.

Classify the sentence as either:
- Premise
- Claim

Examples:

"""

    for _, row in examples.iterrows():

        prompt += f"""
Sentence: {row['text']}
Label: {row['answer']}

"""

    prompt += f"""
Now classify:

Sentence: {query}

Label:
"""

    return prompt