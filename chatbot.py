# chatbot.py
# 把 Colab notebook 的核心邏輯整合到這裡

import re
import json
import copy
import ast
import torch
import numpy as np
import pandas as pd
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ── 路徑設定 ──────────────────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL  = os.path.join(BASE_DIR, "models", "nlu_model", "nlu_model")
SLOT_W_PATH = os.path.join(BASE_DIR, "models", "slot_weights.json")
DST_PATH    = os.path.join(BASE_DIR, "models", "dst_module.py")
KG_PATH     = os.path.join(BASE_DIR, "data", "knowledge_graph_neighbourhood_sentiment.csv")
MODEL_NAME  = "answerdotai/ModernBERT-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CHATBOT] Device: {device}")

# ── 載入 Label Maps ───────────────────────────────────────────
try:
    with open(os.path.join(BASE_MODEL, "label_maps.json")) as f:
        label_maps = json.load(f)
except FileNotFoundError as e:
    print(f"[ERROR] label_maps.json not found: {e}")
    raise

INTENT2ID   = label_maps["intent2id"]
ID2INTENT   = {int(k): v for k, v in label_maps["id2intent"].items()}
SLOT2ID     = label_maps["slot2id"]
ID2SLOT     = {int(k): v for k, v in label_maps["id2slot"].items()}
INTENT_LABELS = list(INTENT2ID.keys())
SLOT_LABELS   = list(SLOT2ID.keys())

# ── 載入 Tokenizer ────────────────────────────────────────────
try:
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_MODEL, "tokenizer"))
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer: {e}")
    raise

# ── 模型定義 ──────────────────────────────────────────────────
class JointNLUModel(nn.Module):
    def __init__(self, model_name, num_intents, num_slots, dropout=0.1):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.dropout    = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head   = nn.Linear(hidden, num_slots)

    def forward(self, input_ids, attention_mask):
        out          = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq          = out.last_hidden_state
        cls_out      = self.dropout(seq[:, 0, :])
        intent_logits = self.intent_head(cls_out)
        slot_logits   = self.slot_head(self.dropout(seq))
        return intent_logits, slot_logits

# ── 載入模型權重 ───────────────────────────────────────────────
try:
    model = JointNLUModel(MODEL_NAME, len(INTENT_LABELS), len(SLOT_LABELS)).to(device)
    model.load_state_dict(
        torch.load(os.path.join(BASE_MODEL, "joint_nlu_weights.pt"), map_location=device)
    )
    model.eval()
    print("✓ NLU 模型載入完成")
except Exception as e:
    print(f"[ERROR] Failed to load NLU model: {e}")
    raise

# ── Trigger 詞典（直接從 notebook 複製）──────────────────────
PRICE_TRIGGERS = {
    1: ["budget-friendly","budget friendly","dirt cheap","cheap eats","cheap","budget","affordable","inexpensive","cheaper"],
    2: ["mid-range","mid range","moderate","decent"],
    3: ["upscale","upscaled","fancy","fancier","bougie","pricey"],
    4: ["fine dining","high-end","high end","splurge"],
}
OUTDOOR_SEATING_TRIGGERS = ["outdoor seating","garden dining","al fresco","patio","terrace","outdoor","outside"]
GOOD_FOR_GROUPS_TRIGGERS = ["good for groups","large parties","large party","big group","big groups","group","groups","party","birthday","family","gathering","celebration","celebratory"]
NOISE_LEVEL_TRIGGERS = {
    "quiet":   ["quiet","peaceful","calm","not too loud"],
    "average": ["average noise","moderate noise"],
    "loud":    ["loud","lively","buzzy","noisy","energetic"],
}
MIN_RATING_TRIGGERS = {
    5: ["5 stars","five star","the best"],
    4: ["4 stars","four star","highly rated","top rated","well reviewed","great reviews","popular spot","crowd favourite","local favourite","worth it"],
    3: ["3 stars","decent","good enough","not bad"],
}
DELIVERY_TRIGGERS = ["delivery","delivers","home delivery","order in","get it delivered","don't want to go out","stay in and eat"]
TAKE_OUT_TRIGGERS = ["takeout","take away","to go","pickup","grab and go","eat at home"]
WIFI_TRIGGERS     = ["wifi","free wifi","good for working","laptop friendly","work remotely","study"]

# ── Tokenisation ──────────────────────────────────────────────
def simple_tokenise(text: str) -> list:
    tokens = text.lower().split()
    tokens = [re.sub(r"[^\w'-]", "", t) for t in tokens]
    return [t for t in tokens if t]

def find_phrase_in_tokens(tokens, phrase):
    phrase_tokens = simple_tokenise(phrase)
    n = len(phrase_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == phrase_tokens:
            return (i, i+n)
    return None

# ── NLU 推理 ──────────────────────────────────────────────────
def predict_intent(text: str) -> str:
    tokens   = simple_tokenise(text)
    encoding = tokenizer(
        tokens, is_split_into_words=True,
        return_tensors="pt", truncation=True,
        max_length=64, padding="max_length",
    ).to(device)
    with torch.no_grad():
        intent_logits, _ = model(encoding["input_ids"], encoding["attention_mask"])
    return ID2INTENT[intent_logits.argmax(dim=-1).item()]

def predict_slots_raw(text: str) -> list:
    tokens   = simple_tokenise(text)
    encoding = tokenizer(
        tokens, is_split_into_words=True,
        return_tensors="pt", truncation=True,
        max_length=64, padding="max_length",
    ).to(device)
    with torch.no_grad():
        _, slot_logits = model(encoding["input_ids"], encoding["attention_mask"])
    slot_ids  = slot_logits.argmax(dim=-1).squeeze().tolist()
    word_ids  = encoding.word_ids()
    result, seen = [], set()
    for pos, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        result.append((tokens[word_idx], ID2SLOT.get(slot_ids[pos], "O")))
    return result

def bio_to_slots(bio_predictions: list) -> dict:
    slots, current_slot, current_value = {}, None, []
    for token, tag in bio_predictions:
        if tag.startswith("B-"):
            if current_slot:
                slots[current_slot] = " ".join(current_value)
            current_slot  = tag[2:]
            current_value = [token]
        elif tag.startswith("I-") and current_slot == tag[2:]:
            current_value.append(token)
        else:
            if current_slot:
                slots[current_slot] = " ".join(current_value)
                current_slot, current_value = None, []
    if current_slot:
        slots[current_slot] = " ".join(current_value)
    return slots

PRICE_NORMALISE = {
    "cheap":1,"budget":1,"affordable":1,"inexpensive":1,"cheaper":1,
    "dirt cheap":1,"cheap eats":1,"budget-friendly":1,"budget friendly":1,
    "moderate":2,"mid-range":2,"mid range":2,"decent":2,
    "upscale":3,"upscaled":3,"fancy":3,"fancier":3,"bougie":3,"pricey":3,
    "fine dining":4,"high-end":4,"high end":4,"splurge":4,
}
BOOLEAN_SLOTS = {"outdoor_seating","good_for_groups"}

def normalise_slots(slots: dict) -> dict:
    normalised = {}
    for slot, value in slots.items():
        v = value.lower().strip()
        if slot == "price":
            normalised[slot] = PRICE_NORMALISE.get(v, v)
        elif slot in BOOLEAN_SLOTS:
            normalised[slot] = True
        else:
            normalised[slot] = v
    return normalised

def build_constraint_dict(text: str) -> dict:
    bio_preds = predict_slots_raw(text)
    raw_slots = bio_to_slots(bio_preds)
    return normalise_slots(raw_slots)

def nlu_pipeline(text: str) -> dict:
    return {
        "intent":      predict_intent(text),
        "constraints": build_constraint_dict(text),
    }

# ── DST ───────────────────────────────────────────────────────
try:
    with open(SLOT_W_PATH) as f:
        SLOT_WEIGHTS = json.load(f)
except FileNotFoundError as e:
    print(f"[ERROR] slot_weights.json not found: {e}")
    raise

# 載入 DialogueStateTracker class
try:
    with open(DST_PATH) as f:
        dst_code = f.read()
    exec(dst_code, globals())
    print("✓ DialogueStateTracker 載入完成")
except FileNotFoundError as e:
    print(f"[ERROR] dst_module.py not found: {e}")
    raise
except Exception as e:
    print(f"[ERROR] Failed to load DialogueStateTracker: {e}")
    raise

# ── Knowledge Graph ───────────────────────────────────────────
try:
    kg = pd.read_csv(KG_PATH)
    print(f"✓ Knowledge graph 載入完成：{len(kg)} 間餐廳")
except FileNotFoundError as e:
    print(f"[ERROR] Knowledge graph CSV not found: {e}")
    raise

# ── Dialogue Manager（直接從 notebook 的 Cell 111 複製）──────
FILTERABLE_SLOTS  = {"cuisine","location","price","dietary","outdoor_seating",
                     "good_for_groups","noise_level","wifi","take_out","delivery","min_rating"}
RANKING_ONLY_SLOTS = {"ambience"}

def filter_kg(kg_df, constraints):
    mask = pd.Series(True, index=kg_df.index)
    for slot, value in constraints.items():
        if slot in RANKING_ONLY_SLOTS:
            continue
        if slot == "cuisine":
            mask &= kg_df["cuisines"].apply(
                lambda x: value.lower() in [c.lower().strip() for c in ast.literal_eval(x)]
                if pd.notna(x) and x != "[]" else False
            )
        elif slot == "price":
            try:
                mask &= kg_df["price_range"] == int(value)
            except Exception:
                pass
        elif slot == "location":
            # 用 address 或 city 列進行模糊匹配
            location_lower = value.lower().strip()
            mask &= (kg_df["address"].str.lower().str.contains(location_lower, na=False) |
                    kg_df["city"].str.lower().str.contains(location_lower, na=False))
        elif slot == "outdoor_seating":
            mask &= kg_df["outdoor_seating"] == True
        elif slot == "good_for_groups":
            mask &= kg_df["good_for_groups"] == True
        elif slot == "min_rating":
            try:
                mask &= kg_df["stars"] >= float(value)
            except Exception:
                pass
    return kg_df[mask].copy()

def match_restaurants(kg_df, constraints, top_n=5):
    filtered = filter_kg(kg_df, constraints)
    if len(filtered) == 0:
        return pd.DataFrame()
    filtered = filtered.sort_values("stars", ascending=False)
    return filtered.head(top_n)

class DialogueManager:
    def __init__(self, dst, kg_df, slot_weights,
                 min_completeness=0.40, max_candidates=15, min_candidates=1):
        self.dst             = dst
        self.kg              = kg_df
        self.slot_weights    = slot_weights
        self.min_completeness = min_completeness
        self.max_candidates  = max_candidates
        self.min_candidates  = min_candidates
        self.recommendations = None
        self.asked_slots     = []
        self.last_action     = None
        self.last_slot_asked = None

    def reset(self):
        self.dst.reset() if hasattr(self.dst, 'reset') else None
        self.recommendations = None
        self.asked_slots     = []
        self.last_action     = None
        self.last_slot_asked = None

    def process_turn(self, nlu_output: dict, raw_text: str = "") -> dict:
        intent      = nlu_output.get("intent", "")
        constraints = nlu_output.get("constraints", {})

        # 「no preference」處理
        if intent == "reject" and self.last_action == "ask" and not constraints:
            intent = "find_restaurant"

        self.dst.update({"intent": intent, "constraints": constraints})
        state        = self.dst.get_filled_slots()
        completeness = self.dst.completeness_score() if hasattr(self.dst, 'completeness_score') else 0.5
        candidates   = match_restaurants(self.kg, state)

        if intent == "accept":
            self.last_action = "end_accept"
            return {"action": "end_accept", "message": "Great! Enjoy your meal! 🍽️",
                    "n_candidates": len(candidates), "completeness": completeness}

        if len(candidates) >= self.min_candidates and completeness >= self.min_completeness:
            self.recommendations = candidates
            self.last_action     = "recommend"
            msg = self._format_recommendations(candidates)
            return {"action": "recommend", "message": msg,
                    "n_candidates": len(candidates), "completeness": completeness}

        # 問下一個 slot
        missing = self.dst.get_highest_priority_missing() if hasattr(self.dst, 'get_highest_priority_missing') else None
        if missing:
            slot = missing[0]
            self.last_slot_asked = slot
            self.last_action     = "ask"
            q = CLARIFYING_QUESTIONS.get(slot, f"Can you tell me more about your {slot} preference?")
            return {"action": "ask", "slot_asked": slot, "message": q,
                    "n_candidates": len(candidates), "completeness": completeness}

        # 找不到更多 slot，就推薦
        self.recommendations = candidates
        self.last_action     = "recommend"
        msg = self._format_recommendations(candidates) if len(candidates) > 0 else NO_RESULTS_MESSAGE
        return {"action": "recommend", "message": msg,
                "n_candidates": len(candidates), "completeness": completeness}

    def _format_recommendations(self, candidates):
        if len(candidates) == 0:
            return NO_RESULTS_MESSAGE
        lines = ["Here are some restaurants I'd recommend:\n"]
        for i, (_, row) in enumerate(candidates.head(3).iterrows(), 1):
            name   = row.get("name", "Unknown")
            stars  = row.get("stars", "?")
            area   = row.get("address", row.get("city", "Philadelphia"))
            lines.append(f"{i}. *{name}* ⭐ {stars} — {area}")
        lines.append("\nWould you like to go with one of these?")
        return "\n".join(lines)

CLARIFYING_QUESTIONS = {
    "cuisine":         "What type of cuisine are you in the mood for? E.g. Italian, Japanese, Mexican.",
    "location":        "Which area of Philadelphia? E.g. Fishtown, Rittenhouse, Old City.",
    "price":           "What's your budget? Cheap, mid-range, upscale, or fine dining?",
    "dietary":         "Any dietary requirements? E.g. vegetarian, vegan, halal, gluten-free.",
    "ambience":        "What kind of vibe? Casual, romantic, quiet, or lively?",
    "outdoor_seating": "Would you like somewhere with outdoor seating?",
    "good_for_groups": "Is this for a group outing?",
    "noise_level":     "Preference on noise level? Quiet, average, or lively?",
    "wifi":            "Do you need the restaurant to have WiFi?",
    "take_out":        "Do you need takeout options?",
    "delivery":        "Do you need delivery?",
    "min_rating":      "Do you have a minimum star rating in mind?",
}
NO_RESULTS_MESSAGE = (
    "I couldn't find any restaurants matching all your criteria. "
    "Would you like to relax some requirements?"
)

# ── 全局 Dialogue Manager 實例 ────────────────────────────────
# 每個 WhatsApp 用戶有自己的 session
_sessions: dict = {}

def get_session(user_id: str) -> DialogueManager:
    """為每個用戶維護獨立的對話狀態"""
    if user_id not in _sessions:
        dst_instance = DialogueStateTracker(SLOT_WEIGHTS)
        _sessions[user_id] = DialogueManager(dst_instance, kg, SLOT_WEIGHTS)
    return _sessions[user_id]

NOISE_LEVEL_KEYWORDS = {
    "loud":    ["loud", "noisy", "lively", "energetic", "buzzy"],
    "quiet":   ["quiet", "peaceful", "calm"],
    "average": ["average", "moderate"],
}
SKIP_PHRASES = ["no preference", "skip", "don't mind", "dont mind", "any", "doesn't matter", "doesnt matter", "no opinion"]

def _extract_noise_level(text: str):
    """Keyword-match noise level from text. Returns matched value or None."""
    lowered = text.lower()
    for value, keywords in NOISE_LEVEL_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return value
    return None

def _is_skip(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in SKIP_PHRASES)

def handle_message(user_id: str, text: str) -> str:
    """主入口：輸入用戶 ID 和訊息，返回 chatbot 回覆"""
    try:
        dm = get_session(user_id)
        print(f"[DEBUG] last_slot_asked: {dm.last_slot_asked}, text: {text}")
        nlu_result = nlu_pipeline(text)

        # ── noise_level 後處理層 ───────────────────────────────
        if dm.last_slot_asked == "noise_level":
            if _is_skip(text):
                # 強制跳過：把 intent 設為 reject，讓現有 no-preference 邏輯接管
                nlu_result = {"intent": "reject", "constraints": {}}
            else:
                matched = _extract_noise_level(text)
                if matched:
                    nlu_result["constraints"]["noise_level"] = matched
                    nlu_result["intent"] = "find_restaurant"
        # ──────────────────────────────────────────────────────

        response   = dm.process_turn(nlu_result, raw_text=text)

        if response["action"] == "end_accept":
            # 對話結束，下次重新開始
            del _sessions[user_id]

        return response["message"]
    except Exception as e:
        print(f"[ERROR] handle_message: {e}")
        # 提供一個 fallback 回覆，避免 Twilio 收不到回應
        return "Sorry, I encountered an error. Please try again."
