from gliner import GLiNER

# 1. Load GLiNER v2.5
model = GLiNER.from_pretrained("gliner-community/gliner_medium-v2.5", load_tokenizer=True)  # :contentReference[oaicite:1]{index=1}

# 2. Define a comprehensive set of entity types
labels = [
    "person","organization","location","event","work of art","consumer good",
    "language","date","time","quantity","ordinal","cardinal","product",
    "law","nationality","money","percent","facility","title"
]

def extract_all_with_gliner(text: str) -> list[str]:
    preds = model.predict_entities(text, labels, threshold=0.3)
    seen = set()
    ents = []
    for p in preds:
        span = p["text"]
        if span not in seen:
            seen.add(span)
            ents.append(span)
    return ents

# Usage:
sample = "Barack Obama and Elon Musk met at Tesla headquarters in California last Tuesday."
print(extract_all_with_gliner(sample))
# → ['Barack Obama','Elon Musk','Tesla','California','last Tuesday']

# M2

# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# # 1. Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# # 2. Create a NER pipeline with simple aggregation
# ner_pipe = pipeline(
#     "ner",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )

# def extract_all_entities(text: str) -> list[str]:
#     """
#     Runs NER and returns a deduped list of all entity spans.
#     """
#     raw = ner_pipe(text)
#     seen = set()
#     ents = []
#     for ent in raw:
#         span = ent["word"]
#         if span not in seen:
#             seen.add(span)
#             ents.append(span)
#     return ents

# # Usage:
# sample = "Satya Nadella joined Bill Gates at Microsoft HQ in Redmond on Monday."
# print(extract_all_entities(sample))
# # → ['Satya Nadella','Bill Gates','Microsoft','Redmond','Monday']

# M3

# import spacy

# # 1. Load a spaCy model with NER (e.g., transformer‑backbone for best accuracy)
# nlp = spacy.load("en_core_web_trf")

# def extract_spacy_entities(text: str) -> list[str]:
#     doc = nlp(text)
#     seen = set()
#     ents = []
#     for ent in doc.ents:
#         if ent.text not in seen:
#             seen.add(ent.text)
#             ents.append(ent.text)
#     return ents

# # Usage:
# sample = "Google CEO Sundar Pichai visited Paris last Friday."
# print(extract_spacy_entities(sample))