import json

# 1. Load the original generated training data
with open("gliner_train.json", "r") as f:
    dataset = json.load(f)

# 2. Define the exact label names to IGNORE. 
# (These match the labels found in your original Masked column)
labels_to_ignore = {
    "Aadhaar_Number",
    "PAN_Card", 
    "GST", "GSTIN",
    "IFSC_Code", 
    "Phone_Number", 
    "Bank_Account", 
    "Voter_ID", 
    "Passport_Number", 
    "Driving_License", 
    "UDYAM", "Udyam",
    "UAN", 
    "Email_Address", "Email"
}

filtered_dataset = []
removed_count = 0

for data in dataset:
    # Filter the 'ner' list: keep the span only if its label (span[2]) is NOT in our ignore list
    original_ner_count = len(data["ner"])
    filtered_ner = [span for span in data["ner"] if span[2] not in labels_to_ignore]
    
    removed_count += (original_ner_count - len(filtered_ner))
    
    # Update the data with the new filtered list
    data["ner"] = filtered_ner
    filtered_dataset.append(data)

# 3. Save the new filtered dataset
with open("gliner_train_filtered.json", "w") as f:
    json.dump(filtered_dataset, f)

print(f"Dataset successfully filtered.")
print(f"Total annotations removed based on the ignore list: {removed_count}")
print("Saved to 'gliner_train_filtered.json'.")