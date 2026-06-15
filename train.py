import json
from gliner import GLiNER
from gliner.training import TrainingArguments, Trainer

# 1. Load the raw list data
with open("gliner_train_filtered.json", "r") as f:
    raw_data = json.load(f)

train_dataset = raw_data[:4000]
eval_dataset = raw_data[4000:]
print(type(train_dataset))
print(type(train_dataset[0]))
print(train_dataset[0])
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print(type(model.data_processor))
print(model.data_processor)

# 2. Define the exact GLiNER internal collator
gliner_internal_collator = getattr(model.data_processor, "collate_fn", getattr(model.data_processor, "collate_raw_batch", None))

# 3. Create the Wrapper Collator
def safe_gliner_collator(batch):
    # If Hugging Face converted the batch into a dictionary of lists, flip it back!
    if isinstance(batch, dict):
        keys = batch.keys()
        length = len(batch[list(keys)[0]])
        batch = [{k: batch[k][i] for k in keys} for i in range(length)]
        
    # Pass the corrected list of dictionaries to GLiNER
    return gliner_internal_collator(batch)

# 4. Define Training Arguments (Keep remove_unused_columns=False!)
training_args = TrainingArguments(
    output_dir="gliner-pii-model",
    learning_rate=5e-6,
    weight_decay=0.01,
    others_lr=1e-5,               
    others_weight_decay=0.01,     
    warmup_ratio=0.1,
    max_steps=1000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",        
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    remove_unused_columns=False, # Critical for raw lists
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
   
)

# Start training
trainer.train()