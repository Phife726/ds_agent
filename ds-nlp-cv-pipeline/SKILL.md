---
name: ds-nlp-cv-pipeline
description: Assists with Natural Language Processing and Computer Vision tasks including text classification, entity extraction, sentiment analysis, image classification, and transfer learning. Use this skill whenever the user wants to work with text data (TF-IDF, tokenization, text classification, NER, sentiment, topic modeling, embeddings) or image data (image classification, object detection, transfer learning, data augmentation). Trigger when the user says "classify these texts", "analyze sentiment", "extract entities", "build an NLP pipeline", "classify images", "use a pretrained model", "transfer learning", "process text data", "build a text classifier", or drops text/image data and asks for predictions. Also trigger for questions about choosing between bag-of-words, TF-IDF, word embeddings, or transformer models, or when the user needs help with Hugging Face, spaCy, or torchvision.
---

# NLP & Computer Vision Pipeline Skill

This skill helps you build pipelines for text and image data — two domains where raw inputs aren't numeric and require specialized preprocessing before any model can use them. The core challenge in both is the same: converting unstructured data (words, pixels) into meaningful numerical representations.

## Part 1: Natural Language Processing

### Choosing a Text Representation

The right representation depends on your task and data size. Here's the decision framework:

**Bag of Words / TF-IDF** — Start here for most classification tasks.
- Works well with <50k documents
- Pairs naturally with scikit-learn classifiers
- TF-IDF is almost always better than raw counts (it downweights common words)
- Limitation: ignores word order entirely

**Word Embeddings (Word2Vec, GloVe, FastText)** — When semantic similarity matters.
- Captures meaning: "king" - "man" + "woman" ≈ "queen"
- Pre-trained versions available (no need to train your own)
- Average embeddings across a document for a simple document vector
- Better than TF-IDF when vocabulary overlap between documents is low

**Transformer Models (BERT, RoBERTa, etc.)** — When you need state-of-the-art performance.
- Captures context: "bank" means different things in different sentences
- Requires more compute (GPU strongly recommended)
- Fine-tuning on your data usually outperforms everything else
- Use Hugging Face transformers library

### Text Classification with TF-IDF + scikit-learn

This is the workhorse pipeline for most text classification tasks.

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,      # vocabulary size cap
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2,                # ignore very rare terms
        max_df=0.95,             # ignore terms in >95% of docs
        strip_accents='unicode',
        stop_words='english'
    )),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Cross-validate
cv_scores = cross_val_score(text_pipeline, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Evaluate on test set
text_pipeline.fit(X_train, y_train)
y_pred = text_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

**Tuning tips:**
- `ngram_range=(1, 2)` adds bigrams, which helps capture phrases like "not good"
- `min_df=2` removes typos and one-off terms that add noise
- `max_df=0.95` removes words so common they carry no signal
- Try `sublinear_tf=True` for logarithmic term frequency scaling

### Named Entity Recognition with spaCy

```python
import spacy

nlp = spacy.load('en_core_web_sm')  # or en_core_web_trf for transformer-based

doc = nlp("Apple is looking at buying a startup in San Francisco for $1 billion.")

for ent in doc.ents:
    print(f"  {ent.text:20s} → {ent.label_:10s}")
# Apple                → ORG
# San Francisco        → GPE
# $1 billion           → MONEY
```

For custom entity types, use spaCy's `EntityRuler` or fine-tune the NER model on your labeled data.

### Text Classification with Hugging Face Transformers

When TF-IDF isn't cutting it and you need contextual understanding:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This product is absolutely fantastic!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

**For fine-tuning on custom data:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(label_set)
)

def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
trainer.train()
```

### Text Preprocessing Checklist

Before feeding text to any model, consider these steps (not all are always needed):
1. **Lowercasing** — almost always (TF-IDF handles this with `lowercase=True`)
2. **Remove HTML tags** — if scraping web data
3. **Handle contractions** — "don't" → "do not" (optional, helps some models)
4. **Remove or replace URLs and emails** — unless they're features
5. **Tokenization** — splitting text into words/subwords
6. **Stop word removal** — removes "the", "is", etc. Good for TF-IDF, bad for transformers
7. **Lemmatization** — "running" → "run". Better than stemming but slower. Use spaCy.

Important: Transformer models handle their own tokenization. Don't apply aggressive text cleaning before feeding text to BERT — it was trained on natural text.

## Part 2: Computer Vision

### Transfer Learning (The Default Approach)

Training a CNN from scratch requires millions of images. Transfer learning uses a model pre-trained on ImageNet and adapts it to your task. This is almost always the right approach unless you have an extremely specialized domain.

```python
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn

# Load pretrained model and replace final layer
model = models.resnet50(pretrained=True)

# Freeze all layers except the final classifier
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your number of classes
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define transforms (match what the pretrained model expects)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Data Augmentation

Small image datasets benefit enormously from augmentation — creating variations of existing images during training.

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Only augment training data. Validation and test data should use deterministic transforms (resize + center crop only).

### Image Classification with Hugging Face

For quick prototyping without writing PyTorch training loops:

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("path/to/image.jpg")
for pred in result:
    print(f"  {pred['label']}: {pred['score']:.3f}")
```

### When to Use What

| Scenario | Approach |
|---|---|
| < 100 images per class | Transfer learning with frozen backbone, only train final layer |
| 100–1000 images per class | Transfer learning with fine-tuning last few layers |
| 1000+ images per class | Fine-tune the entire model or train from scratch |
| Quick prototype | Hugging Face pipeline with pretrained model |
| Production deployment | PyTorch/TensorFlow with proper training loop, validation, checkpointing |

## Output Format

When building NLP or CV pipelines, always include:
1. Data exploration (class distribution, sample examples, text length distribution for NLP)
2. Justification for the chosen representation/architecture
3. The complete pipeline with preprocessing
4. Evaluation metrics appropriate to the task
5. Error analysis — look at the worst predictions to understand failure modes
6. Suggestions for improvement (more data, better augmentation, different model)
