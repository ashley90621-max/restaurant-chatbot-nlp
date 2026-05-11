# Restaurant Recommendation Chatbot

**Course**: MSIN0243 Natural Language Processing | UCL School of Management  
**Author**: Kuei-Chun Liu (Group Project)  
**Tools**: Python · PyTorch · HuggingFace Transformers · ModernBERT · Pandas

---

## Overview

A task-oriented dialogue system that recommends Philadelphia restaurants through natural conversation. The system combines a fine-tuned NLU model with a Knowledge Graph and Dialogue State Tracker to progressively narrow down restaurant options based on user preferences.

## System Architecture

User Input
↓
NLU Pipeline (ModernBERT)
├── Intent Classification (find_restaurant / accept / reject)
└── Slot Filling (cuisine, location, price, dietary, ambience...)
↓
Dialogue State Tracker (DST)
└── Tracks filled slots, computes completeness score
↓
Dialogue Manager
├── Filter Knowledge Graph by constraints
├── Ask clarifying questions if completeness < 40%
└── Recommend top-3 restaurants when ready

## Key Features

- 🤖 **Joint NLU Model**: Fine-tuned ModernBERT for simultaneous intent detection and slot filling
- 🗂️ **Knowledge Graph**: 5,854 Philadelphia restaurants with sentiment-enriched attributes
- 💬 **Multi-turn Dialogue**: Progressive slot filling with clarifying questions
- 👥 **Multi-user Sessions**: Independent conversation state per user
- 📱 **WhatsApp Integration**: Designed for Twilio WhatsApp API deployment

## Dataset

- **Source**: Yelp Open Dataset (Philadelphia subset)
- **Scale**: 5,854 restaurants · 687,499 reviews
- **Features**: Cuisine, price range, location, outdoor seating, noise level, sentiment scores

## Example Conversation

User: "I want Italian food"
→ Intent: find_restaurant | Slots: {cuisine: italian}
→ System: "Which area of Philadelphia? E.g. Fishtown, Rittenhouse, Old City."
User: "somewhere in fishtown"
→ Slots: {cuisine: italian, location: fishtown}
→ System: "Here are my top picks:
1. Il Ghiottone ⭐ 4.5 — Fishtown
2. Bistro Sorriso ⭐ 4.5 — Northern Liberties
3. Medusa Pizzeria ⭐ 4.0 — Fishtown"

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Place model files in `models/nlu_model/` and data in `data/`
3. Run the chatbot:
```bash
python app.py
```

> ⚠️ Model weights (~570MB) and dataset files are not included due to size limits.  
> The NLU model was fine-tuned on synthetic training data generated for this project.
