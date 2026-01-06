#Finetuned on smaller dataset. new version proper finetunned model with smalll custom dataset.

!pip install -q transformers datasets peft accelerate scikit-learn torch

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import json

print("="*70)
print("🚀 CELL 1: FINE-TUNING CODE")
print("="*70)

# ============================================
# PARAMETERS
# ============================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ft_model_dir = "./tinyllama_sql_finetuned"  # Local directory
train_epochs = 1
batch_size = 2
learning_rate = 2e-4
max_length = 512
validation_split = 0.05

print(f"\n📋 Configuration:")
print(f"   Model: {model_name}")
print(f"   Save Directory: {ft_model_dir}")
print(f"   Epochs: {train_epochs}, Batch Size: {batch_size}\n")

# ============================================
# LOAD TOKENIZER & MODEL
# ============================================
print("📥 Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("✅ Base model loaded!\n")

# ============================================
# CREATE DATASET
# ============================================
print("📊 Creating dataset...")
data_samples = [
    {"question": "List all users from the database", "sql": "SELECT * FROM users;"},
    {"question": "Count total orders", "sql": "SELECT COUNT(*) FROM orders;"},
    {"question": "Get customer names where age > 25", "sql": "SELECT name FROM customers WHERE age > 25;"},
    {"question": "Find all products with price > 100", "sql": "SELECT * FROM products WHERE price > 100;"},
    {"question": "Delete orders older than 2022", "sql": "DELETE FROM orders WHERE order_date < '2022-01-01';"},
    {"question": "Update product stock to 50 for id 10", "sql": "UPDATE products SET stock=50 WHERE id=10;"},
    {"question": "Select emails from users", "sql": "SELECT email FROM users;"},
    {"question": "Get orders sorted by date descending", "sql": "SELECT * FROM orders ORDER BY order_date DESC;"},
    {"question": "Find customers from city Lahore", "sql": "SELECT * FROM customers WHERE city='Lahore';"},
    {"question": "Insert new user John into users", "sql": "INSERT INTO users (name) VALUES ('John');"}
]

random.shuffle(data_samples)
split_idx = int(len(data_samples) * (1 - validation_split))
train_dataset = Dataset.from_list(data_samples[:split_idx])
val_dataset = Dataset.from_list(data_samples[split_idx:])

# Preprocessing function
def preprocess(example):
    input_text = f"Translate to SQL: {example['question']}"
    output_text = example['sql']
    input_ids = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length).input_ids
    labels = tokenizer(output_text, truncation=True, padding="max_length", max_length=max_length).input_ids
    return {"input_ids": input_ids, "labels": labels}

train_dataset = train_dataset.map(preprocess, remove_columns=['question', 'sql'])
val_dataset = val_dataset.map(preprocess, remove_columns=['question', 'sql'])

train_dataset.set_format("torch", columns=['input_ids', 'labels'])
val_dataset.set_format("torch", columns=['input_ids', 'labels'])

print(f"✅ Dataset ready - Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

# ============================================
# LORA SETUP
# ============================================
print("🔧 Setting up LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
print("📊 LoRA Configuration:")
model.print_trainable_parameters()
print()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ============================================
# TRAINING ARGUMENTS
# ============================================
training_args = TrainingArguments(
    output_dir=ft_model_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=train_epochs,
    learning_rate=learning_rate,
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    logging_steps=1,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    remove_unused_columns=False
)

# ============================================
# METRICS FUNCTION
# ============================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels.flatten(),
        valid_predictions.flatten(),
        average='micro',
        zero_division=0
    )
    acc = accuracy_score(valid_labels.flatten(), valid_predictions.flatten())

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================
# TRAINER
# ============================================
print("🏋️ Starting training...\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ============================================
# SAVE FINE-TUNED MODEL
# ============================================
print("\n💾 Saving fine-tuned model to local directory...")
os.makedirs(ft_model_dir, exist_ok=True)

# Save LoRA weights
model.save_pretrained(ft_model_dir, safe_serialization=True)
print(f"✅ LoRA weights saved")

# Save LoRA config
lora_config.save_pretrained(ft_model_dir)
print(f"✅ LoRA config saved")

# Save tokenizer
tokenizer.save_pretrained(ft_model_dir)
print(f"✅ Tokenizer saved")

# Verify files
files = os.listdir(ft_model_dir)
print(f"\n✅ Files in directory: {files}\n")

print("="*70)
print("✅ FINE-TUNING COMPLETE!")
print("="*70)
print(f"\n📁 Model saved to: {ft_model_dir}")
print(f"📊 Best Loss: {trainer.state.best_metric:.4f}")
print("\n🎯 Next Step: Run CELL 2 to load this model and use Gradio!")
print("="*70)

#after finetunnig run this code below

#Finetuned on smaller dataset.

!pip install -q transformers peft torch gradio

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import json

print("="*70)
print("🚀 CELL 2: LOAD FINE-TUNED MODEL + GRADIO INTERFACE")
print("="*70)

# ============================================
# PARAMETERS
# ============================================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ft_model_dir = "./tinyllama_sql_finetuned"  # Same directory from CELL 1

print(f"\n📋 Configuration:")
print(f"   Base Model: {model_name}")
print(f"   Fine-tuned Model Dir: {ft_model_dir}\n")

# ============================================
# VERIFY MODEL EXISTS
# ============================================
print("🔍 Checking fine-tuned model files...")
if not os.path.exists(ft_model_dir):
    print(f"❌ ERROR: Directory '{ft_model_dir}' not found!")
    print(f"   Please run CELL 1 first to fine-tune the model!")
    raise FileNotFoundError(f"Model directory not found: {ft_model_dir}")

files = os.listdir(ft_model_dir)
print(f"✅ Files found: {files}\n")

required_files = ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer.model']
for f in required_files:
    if f not in files:
        print(f"⚠️ Warning: {f} not found")

# ============================================
# LOAD TRANSLATOR
# ============================================
print("🌍 Loading Translator...")
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-mul-en",
    device=0 if torch.cuda.is_available() else -1
)
print("✅ Translator loaded!\n")

# ============================================
# LOAD BASE MODEL
# ============================================
print("📥 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
print("✅ Base model loaded!")

# ============================================
# LOAD TOKENIZER
# ============================================
print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ft_model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Tokenizer loaded!")

# ============================================
# LOAD LORA WEIGHTS
# ============================================
print("📥 Loading LoRA weights...")
try:
    model = PeftModel.from_pretrained(base_model, ft_model_dir, is_trainable=False)
    print("✅ LoRA weights loaded!")

    # Merge LoRA with base model
    print("🔄 Merging LoRA with base model...")
    model = model.merge_and_unload()
    print("✅ Model merged!\n")
except Exception as e:
    print(f"❌ ERROR loading LoRA: {e}")
    raise

print("="*70)
print("✅ FINE-TUNED MODEL READY!")
print("="*70 + "\n")

# ============================================
# SQL GENERATION FUNCTION
# ============================================
def generate_sql(user_input, context=""):
    """Generate SQL from natural language question"""
    try:
        if not user_input.strip():
            return "❌ Please enter a question!"

        # Translate if needed
        translated_text = translator(user_input)[0]["translation_text"]
        print(f"🔤 Translated: {translated_text}")

        prompt = f"""SQL Generator
====================
Context: {context}
Question: {translated_text}
SQL:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output_tokens = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

        result = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        if "SQL:" in result:
            result = result.split("SQL:")[-1].strip()

        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# GRADIO INTERFACE
# ============================================
print("🎨 Creating Gradio Interface...\n")

def sql_generator_app(question, context):
    """Gradio app function"""
    sql_result = generate_sql(question, context)
    return f"💡 **Generated SQL:**\n\n```sql\n{sql_result}\n```"

# Create Gradio Interface
with gr.Blocks(title="SQL Query Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🖥️ SQL Query Generator

    **Fine-Tuned TinyLlama Model with LoRA**

    Convert natural language questions to SQL queries in multiple languages!

    🌍 **Supported Languages:** English, Urdu, Hindi, and more!
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Input")
            question = gr.Textbox(
                placeholder="e.g., Mukhtar tamaami customers ko dikha do OR Show me all customers",
                label="Natural Language Question",
                lines=3
            )
            context = gr.Textbox(
                placeholder="Optional: Database table descriptions",
                label="Database Context",
                lines=2
            )

    with gr.Row():
        with gr.Column():
            submit_btn = gr.Button("🚀 Generate SQL", variant="primary", size="lg")
        with gr.Column():
            clear_btn = gr.Button("🔄 Clear", size="lg")

    output = gr.Markdown(label="Output")

    submit_btn.click(
        fn=sql_generator_app,
        inputs=[question, context],
        outputs=output
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question, context, output]
    )

    gr.Examples(
        examples=[
            ["Show me all users", ""],
            ["Kitne orders hain", ""],
            ["Find customers from Lahore", "Database has customers table with city column"],
            ["Products ke saath price 100 se zyada", ""],
            ["Delete old records", "Records older than 2022"],
        ],
        inputs=[question, context],
    )

    gr.Markdown("""
    ---
    ### 📊 Model Information

    - **Base Model:** TinyLlama 1.1B
    - **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
    - **Trainable Parameters:** 0.1% (1.1M out of 1.1B)
    - **Translator:** Helsinki-NLP Multilingual (mul-en)
    - **Framework:** Transformers + Gradio + PyTorch

    **Status:** ✅ Ready to use!
    """)

# ============================================
# LAUNCH GRADIO
# ============================================
print("="*70)
print("🎉 LAUNCHING GRADIO INTERFACE!")
print("="*70 + "\n")
demo.launch(share=True)

