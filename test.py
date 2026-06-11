from huggingface_hub import login
from transformers import pipeline

# 1. Force the authentication at the environment level
my_token = "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" # <-- PASTE YOUR REAL TOKEN HERE
login(token=my_token)

# 2. Attempt to download/load the model
print("Authentication successful. Attempting to download IndicNER...")
try:
    nlp = pipeline("ner", model="ai4bharat/IndicNER", aggregation_strategy="simple")
    print("\n✅ SUCCESS! The model downloaded and loaded perfectly.")
    
    # Quick test
    res = nlp("My name is Ashish Kumar.")
    print("Test Output:", res)
    
except Exception as e:
    print("\n❌ FAILED. Here is the error:")
    print(e)