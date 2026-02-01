import re
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import spacy


# =========================
# Labels (wie Training/Eval)
# =========================
LABELS = [
    "TARGET_ENTITY_ORG",
    "EVIDENCE_TARGET_ENTITY",
    "UBO_PERSON",
    "EVIDENCE_UBO",
    "UBO_STELLER_ORG",
    "UBO_STELLER_PERSON",
    "EVIDENCE_UBO_STELLER",
    "PERSON_NOT_UBO",
    "EVIDENCE_NOT_UBO",
    "UBO_CANDIDATE",
    "EVIDENCE_UBO_CANDIDATE",
]

PERSON_LIKE_LABELS = {"UBO_PERSON", "UBO_CANDIDATE", "UBO_STELLER_PERSON", "PERSON_NOT_UBO"}
EVIDENCE_LABELS = {
    "EVIDENCE_TARGET_ENTITY",
    "EVIDENCE_UBO",
    "EVIDENCE_UBO_STELLER",
    "EVIDENCE_NOT_UBO",
    "EVIDENCE_UBO_CANDIDATE",
}

# =========================
# spaCy sentence helpers
# =========================
_NLP = None

def get_nlp(model_name: str):
    global _NLP
    if _NLP is None:
        _NLP = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        if "parser" not in _NLP.pipe_names and "senter" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP

def spacy_sentence_spans(text: str, model_name: str) -> List[Tuple[int, int]]:
    nlp = get_nlp(model_name)
    doc = nlp(text)
    return [(s.start_char, s.end_char) for s in doc.sents]

def sent_id_for_pos(pos: int, sents: List[Tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(sents):
        if s <= pos < e:
            return i
    return -1


# =========================
# Normalise/dedup
# =========================
def normalize_pred(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start": int(p["start"]),
        "end": int(p["end"]),
        "label": str(p["label"]),
        "score": float(p.get("score", 0.0)),
        "text": str(p.get("text", "")),
    }

def dedup_same_span_keep_best(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for p in preds:
        key = (int(p["start"]), int(p["end"]))
        if key not in best or float(p["score"]) > float(best[key]["score"]):
            best[key] = p
    return list(best.values())

def _clean_snippet(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# Deine Regex-Regeln (aus dem Script)
# =========================
_STELLER_EVIDENCE_PATTERNS = [
    r"\beinen\s+solchen\s+im\s+nennbetrag\s+von\b",
    r"\bbenennen\s+jeweils\s+einen\s+stellvertreter\s+des\s+vorsitzenden\s+des\s+aufsichtsrats\b",
    r"\bstellvertreter\b",
    r"\bbenennen\b.*\bstellvertreter\b",
]
_STELLER_EVIDENCE_RE = re.compile("|".join(_STELLER_EVIDENCE_PATTERNS), flags=re.IGNORECASE)

_PERCENT_25_OR_MORE = r"(?:mehr\s+als\s*25|mindestens\s*25|25\s*(?:%?\s*oder\s*mehr)|25)\s*%?"
_TARGETS = r"(?:stammkapital|stimmrechte|stimmrecht|stimmen|geschäftsanteil(?:e)?|anteil(?:e)?)"
_VERBS = r"(?:hält|halten|haelt|verfügt|verfuegt|besitzt|beteiligt)"

_UBO_AT_LEAST_25_RE = re.compile(rf"(?is){_PERCENT_25_OR_MORE}.{{0,2000}}?{_TARGETS}.{{0,2000}}?{_VERBS}")
_UBO_AT_LEAST_25_RE_ALT = re.compile(rf"(?is){_VERBS}.{{0,2000}}?{_PERCENT_25_OR_MORE}.{{0,2000}}?{_TARGETS}")

_CONTROL_RIGHTS_RE = re.compile(
    r"(?is)(?:bei\s+stimmengleichheit|stimme\s+des\s+gesellschafters|"
    r"nur\s+mit\s+zustimmung|zustimmung\s+des\s+gesellschafters|sonderrecht|"
    r"sperrminorität|sperrminoritaet|vetorecht|veto|stimmbindung|stimmrechtsbindung|"
    r"qualifizierte\s+mehrheit|mehrheitserfordernis)"
)

_NEGATION_RE = re.compile(
    r"(?is)\b(nicht|kein|keine|keiner|keinem|keinen|ausgeschlossen|"
    r"nicht\s+wirt(?:schaftlich)?\s+berechtigt|keine\s+wirt(?:schaftlich)?\s+berechtigung)\b"
)


def _ubo_sentence_ids(text: str, sents: List[Tuple[int, int]]) -> set:
    ubo_sents = set()
    for sid, (s0, s1) in enumerate(sents):
        sent_txt = text[s0:s1] or ""
        if _UBO_AT_LEAST_25_RE.search(sent_txt) or _UBO_AT_LEAST_25_RE_ALT.search(sent_txt):
            ubo_sents.add(sid)
            continue
        if _CONTROL_RIGHTS_RE.search(sent_txt):
            ubo_sents.add(sid)
            continue
    return ubo_sents


def apply_steller_person_rule(text: str, preds: List[Dict[str, Any]], sents: List[Tuple[int, int]],
                             char_window: int = 250, require_same_sentence: bool = True) -> List[Dict[str, Any]]:
    person_like = {"UBO_PERSON", "UBO_CANDIDATE", "PERSON_NOT_UBO", "UBO_STELLER_PERSON"}

    sent_has_steller = set()
    for sid, (s0, s1) in enumerate(sents):
        if _STELLER_EVIDENCE_RE.search(text[s0:s1] or ""):
            sent_has_steller.add(sid)

    # Evidence im Satz um-labeln
    new_preds = []
    for p in preds:
        lab = p.get("label")
        if lab in EVIDENCE_LABELS:
            sid = sent_id_for_pos(int(p["start"]), sents)
            snippet = _clean_snippet(text[int(p["start"]):int(p["end"])])
            if (sid in sent_has_steller) or _STELLER_EVIDENCE_RE.search(snippet or ""):
                p2 = dict(p); p2["label"] = "EVIDENCE_UBO_STELLER"
                new_preds.append(p2); continue
        new_preds.append(p)
    preds = new_preds

    anchors = []
    for sid in sent_has_steller:
        s0, s1 = sents[sid]
        anchors.append((s0, s1, sid))

    for p in preds:
        if p.get("label") in EVIDENCE_LABELS:
            snippet = _clean_snippet(text[int(p["start"]):int(p["end"])])
            if _STELLER_EVIDENCE_RE.search(snippet or ""):
                sid = sent_id_for_pos(int(p["start"]), sents)
                anchors.append((int(p["start"]), int(p["end"]), sid))

    if not anchors:
        return preds

    out = []
    for p in preds:
        if p.get("label") not in person_like:
            out.append(p); continue

        ps, pe = int(p["start"]), int(p["end"])
        psid = sent_id_for_pos(ps, sents)

        make_steller = False
        for a0, a1, asid in anchors:
            if require_same_sentence and (psid != asid):
                continue
            dist = max(0, max(ps, a0) - min(pe, a1))
            if dist <= char_window:
                make_steller = True
                break

        if make_steller:
            p2 = dict(p); p2["label"] = "UBO_STELLER_PERSON"
            out.append(p2)
        else:
            out.append(p)

    return out


def apply_ubo_evidence_relabel_from_ubo_sentences(text: str, preds: List[Dict[str, Any]], sents: List[Tuple[int, int]],
                                                 ubo_sents: set, require_same_sentence: bool = True) -> List[Dict[str, Any]]:
    if not ubo_sents:
        return preds

    sent_text_cache = {sid: (text[sents[sid][0]:sents[sid][1]] or "") for sid in ubo_sents}

    out = []
    for p in preds:
        lab = p.get("label")
        sid = sent_id_for_pos(int(p["start"]), sents)

        if require_same_sentence and (sid not in ubo_sents):
            out.append(p); continue

        if lab == "EVIDENCE_UBO_STELLER":
            out.append(p); continue

        if lab == "EVIDENCE_NOT_UBO":
            stxt = sent_text_cache.get(sid, "")
            if not _NEGATION_RE.search(stxt):
                p2 = dict(p); p2["label"] = "EVIDENCE_UBO"
                out.append(p2)
            else:
                out.append(p)
            continue

        if lab in EVIDENCE_LABELS:
            p2 = dict(p); p2["label"] = "EVIDENCE_UBO"
            out.append(p2); continue

        out.append(p)

    return out


def apply_ubo_person_rule_from_ubo_sentences(preds: List[Dict[str, Any]], sents: List[Tuple[int, int]],
                                            ubo_sents: set, require_same_sentence: bool = True) -> List[Dict[str, Any]]:
    if not ubo_sents:
        return preds

    out = []
    for p in preds:
        lab = p.get("label")
        if lab in {"UBO_PERSON", "UBO_CANDIDATE", "PERSON_NOT_UBO", "UBO_STELLER_PERSON"}:
            sid = sent_id_for_pos(int(p["start"]), sents)
            if (not require_same_sentence) or (sid in ubo_sents):
                p2 = dict(p); p2["label"] = "UBO_PERSON"
                out.append(p2); continue
        out.append(p)
    return out


def apply_not_ubo_override_spacy(preds: List[Dict[str, Any]], sents: List[Tuple[int, int]],
                                char_window: int = 250, require_same_sentence: bool = True) -> List[Dict[str, Any]]:
    evid_not = [p for p in preds if p.get("label") == "EVIDENCE_NOT_UBO"]
    if not evid_not:
        return preds

    evid_info = []
    for e in evid_not:
        es, ee = int(e["start"]), int(e["end"])
        evid_info.append((es, ee, sent_id_for_pos(es, sents)))

    new_preds = []
    for p in preds:
        lab = p.get("label")
        if lab not in {"UBO_PERSON", "UBO_CANDIDATE", "UBO_STELLER_PERSON", "PERSON_NOT_UBO"}:
            new_preds.append(p); continue

        ps, pe = int(p["start"]), int(p["end"])
        psid = sent_id_for_pos(ps, sents)

        override = False
        for es, ee, esid in evid_info:
            if require_same_sentence and (psid != esid):
                continue
            dist = max(0, max(ps, es) - min(pe, ee))
            if dist <= char_window:
                override = True
                break

        if override:
            p2 = dict(p); p2["label"] = "PERSON_NOT_UBO"
            new_preds.append(p2)
        else:
            new_preds.append(p)

    return new_preds


def group_by_label(text: str, preds: List[Dict[str, Any]], max_snippets_per_label: int = 5) -> str:
    d = defaultdict(list)
    for p in sorted(preds, key=lambda x: x["score"], reverse=True):
        snip = _clean_snippet(text[p["start"]:p["end"]])
        if snip:
            d[p["label"]].append((snip, p["score"]))

    lines = []
    for label in sorted(d.keys()):
        items = d[label][:max_snippets_per_label]
        shown = [f"{s} ({sc:.2f})" for s, sc in items]
        lines.append(f"{label} ({len(d[label])}): " + " | ".join(shown))
    return "\n".join(lines)


def run_gliner_on_text(model, text: str, threshold: float, spacy_model: str,
                       steller_window: int = 250, steller_same_sentence: bool = True,
                       ubo_same_sentence: bool = True,
                       not_ubo_window: int = 250, not_ubo_same_sentence: bool = True) -> Dict[str, Any]:
    """
    GLiNER predict + deine Postprocessing-Regeln.
    Returns dict with preds + grouped summary.
    """
    sents = spacy_sentence_spans(text, model_name=spacy_model)

    preds_raw = model.predict_entities(text, LABELS, threshold=threshold)
    preds = [normalize_pred(p) for p in preds_raw]
    preds = dedup_same_span_keep_best(preds)

    preds = apply_steller_person_rule(text, preds, sents, char_window=steller_window, require_same_sentence=steller_same_sentence)
    ubo_sents = _ubo_sentence_ids(text, sents)
    preds = apply_ubo_evidence_relabel_from_ubo_sentences(text, preds, sents, ubo_sents, require_same_sentence=ubo_same_sentence)
    preds = apply_ubo_person_rule_from_ubo_sentences(preds, sents, ubo_sents, require_same_sentence=ubo_same_sentence)
    preds = apply_not_ubo_override_spacy(preds, sents, char_window=not_ubo_window, require_same_sentence=not_ubo_same_sentence)

    summary = group_by_label(text, preds, max_snippets_per_label=5)

    return {
        "threshold": threshold,
        "num_preds": len(preds),
        "preds": sorted(preds, key=lambda p: (p["start"], p["end"], p["label"])),
        "summary_grouped": summary,
    }
