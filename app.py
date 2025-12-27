import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import gradio as gr

model_dir = "ner_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
model.eval()

id2label = {
    0: 'B-art', 1: 'B-eve', 2: 'B-geo', 3: 'B-gpe', 4: 'B-nat', 5: 'B-org',
    6: 'B-per', 7: 'B-tim', 8: 'I-art', 9: 'I-eve', 10: 'I-geo', 11: 'I-gpe',
    12: 'I-nat', 13: 'I-org', 14: 'I-per', 15: 'I-tim', 16: 'O'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ner_predict(text):
    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    word_ids = encoding.word_ids(batch_index=0)

    entities = []
    current_entity = None
    current_label = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        word = words[word_idx]
        label_id = int(predictions[idx])
        label = id2label[label_id]

        if label == "O":
            if current_entity is not None:
                entities.append({"text": " ".join(current_entity), "label": current_label})
                current_entity = None
                current_label = None
            continue

        tag, ent_type = label.split("-")
        if tag == "B":
            if current_entity is not None:
                entities.append({"text": " ".join(current_entity), "label": current_label})
            current_entity = [word]
            current_label = ent_type
        elif tag == "I":
            if current_entity is not None and current_label == ent_type:
                current_entity.append(word)
            else:
                current_entity = [word]
                current_label = ent_type

    if current_entity is not None:
        entities.append({"text": " ".join(current_entity), "label": current_label})

    if not entities:
        return "No entities found."

    lines = [f"{e['text']} -> {e['label']}" for e in entities]
    return "\n".join(lines)

demo = gr.Interface(
    fn=ner_predict,
    inputs=gr.Textbox(lines=3, label="Input text"),
    outputs=gr.Textbox(label="Detected entities"),
    title="BERT NER Demo",
    description="Fine-tuned BERT NER model (geo, gpe, org, per, tim, etc.)."
)

if __name__ == "__main__":
    demo.launch()
