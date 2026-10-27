import json
import os
import re

SYSTEM = """You are a clinical scribe. Read the consultation transcript and return ONLY a JSON object.
Rules:
- Use only what is explicitly said. Never invent or infer. Use "" or [] when not stated.
- symptoms.positive = symptoms the patient has; symptoms.negative = symptoms explicitly denied.
- assessment = diagnoses/differentials as stated by the clinician only.
- chief_complaint in the patient's own terms.
- evidence = list of {"field": <top-level key>, "quote": <verbatim words from transcript, max 12 words>}.
- clinical_summary = 2-3 concise sentences suitable for a medical record.
Schema:
{"patient":{"name":"","age":"","sex":"","identifiers":[]},"chief_complaint":"",
"hpi":{"onset":"","duration":"","progression":"","aggravating":[],"relieving":[]},
"symptoms":{"positive":[],"negative":[]},"past_medical_history":[],
"medications":[{"name":"","dose":"","frequency":"","adherence":""}],"allergies":[],
"observations":{"vitals":[],"exam":[]},"assessment":[],
"plan":{"investigations":[],"prescriptions":[],"advice":[],"follow_up":""},
"clinical_summary":"","evidence":[]}"""


def empty():
    return {"patient": {"name": "", "age": "", "sex": "", "identifiers": []}, "chief_complaint": "",
            "hpi": {"onset": "", "duration": "", "progression": "", "aggravating": [], "relieving": []},
            "symptoms": {"positive": [], "negative": []}, "past_medical_history": [],
            "medications": [], "allergies": [],
            "observations": {"vitals": [], "exam": []}, "assessment": [],
            "plan": {"investigations": [], "prescriptions": [], "advice": [], "follow_up": ""},
            "clinical_summary": "", "evidence": []}


def _merge(default, got):
    """Force `got` into the shape of `default`."""
    if isinstance(default, dict):
        got = got if isinstance(got, dict) else {}
        return {k: _merge(v, got.get(k)) for k, v in default.items()}
    if isinstance(default, list):
        return got if isinstance(got, list) else ([got] if got else [])
    return got if isinstance(got, str) else ("" if got is None else str(got))


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _ground(evidence, transcript):
    t = _norm(transcript)
    kept = [e for e in evidence if isinstance(e, dict) and e.get("quote") and _norm(e["quote"]) in t]
    return kept, len(evidence)


def _json_from(text):
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def _claude(t):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=2500, temperature=0,
        system=SYSTEM, messages=[{"role": "user", "content": "TRANSCRIPT:\n" + t}])
    return _json_from(r.content[0].text)


def _openai(t):
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "TRANSCRIPT:\n" + t}])
    return json.loads(r.choices[0].message.content)


SYMPTOMS = ["fever", "sore throat", "cough", "headache", "chest pain", "shortness of breath", "nausea",
            "vomiting", "diarrhea", "diarrhoea", "fatigue", "dizziness", "abdominal pain", "stomach pain",
            "body ache", "rash", "runny nose", "swelling", "back pain", "loss of appetite", "chills",
            "palpitations", "constipation", "joint pain"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}
ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
TESTS = ["throat swab", "cbc", "complete blood count", "x-ray", "xray", "ecg", "ultrasound", "mri", "ct scan",
         "urine test", "blood test", "hba1c", "culture"]


def _heuristic(t):
    d = empty()
    ev = d["evidence"]
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", t) if s.strip()]
    low = t.lower()

    def add(field, quote):
        if quote:
            ev.append({"field": field, "quote": quote.strip()})

    m = re.search(r"(?i:i'm|i am|my name is|name is|this is)\s+(?i:dr\.?\s+)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", t)
    if m:
        d["patient"]["name"] = m.group(1); add("patient", m.group(0))
    m = re.search(r"(\d{1,3})[- ]?(?:years?[- ]old|yo\b|y/o)", low)
    if m:
        d["patient"]["age"] = m.group(1); add("patient", m.group(0))
    else:
        m = re.search(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty)(?:[- ](one|two|three|four|five|six|seven|eight|nine))?\s+years?\s+old", low)
        if m:
            d["patient"]["age"] = str(TENS[m.group(1)] + ONES.get(m.group(2), 0)); add("patient", m.group(0))
    m = re.search(r"\b(male|female|man|woman)\b", low)
    if m:
        d["patient"]["sex"] = "Male" if m.group(1) in ("male", "man") else "Female"; add("patient", m.group(0))

    for s in SYMPTOMS:
        for m in re.finditer(re.escape(s), low):
            before = low[max(0, m.start() - 28): m.start()]
            before = re.split(r"[.?!;]", before)[-1]
            neg = re.search(r"\b(no|denies|without|not|never)\b", before)
            bucket = d["symptoms"]["negative" if neg else "positive"]
            if s not in bucket:
                bucket.append(s)
            add("symptoms", t[m.start():m.end()])
            break
    pos = d["symptoms"]["positive"]
    if pos:
        d["chief_complaint"] = ", ".join(pos[:2]); add("chief_complaint", pos[0])

    m = re.search(r"(?:for|since)\s+(?:the\s+past\s+)?((?:\d+|a|one|two|three|four|five|six|seven|a few|several)\s+(?:hours?|days?|weeks?|months?|years?))", low)
    if m:
        d["hpi"]["duration"] = m.group(1); add("hpi", m.group(0))
    for s in sents:
        sl = s.lower()
        if re.search(r"\b(worse|worsen|worsens|aggravat)", sl):
            d["hpi"]["aggravating"].append(s); add("hpi", s)
        if re.search(r"\b(helps|relieves|relief|improves|better)\b", sl) and not re.search(r"follow", sl):
            d["hpi"]["relieving"].append(s); add("hpi", s)
        if re.search(r"diabet|hypertens|asthma|thyroid|surgery|removed|hospitali[sz]ed|cholesterol|heart disease", sl) \
                and not re.search(r"prescrib|order|advise", sl):
            d["past_medical_history"].append(s); add("past_medical_history", s)
        if re.search(r"\b(drink|rest|avoid|fluids|diet|exercise)\b", sl):
            d["plan"]["advice"].append(s); add("plan", s)
        if re.search(r"\b(throat|tonsil|chest|lungs?|abdomen|heart)\b", sl) and \
                re.search(r"\b(red|swollen|clear|tender|wheez|crackles|normal|enlarged)", sl) and "?" not in s:
            d["observations"]["exam"].append(s); add("observations", s)

    for m in re.finditer(r"([A-Za-z]{4,})\s+(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?))(?:\s+((?:once|twice|thrice|\d+ times)\s+(?:a day|daily)|daily|at night))?", t):
        ctx = t[max(0, m.start() - 60): m.start()].lower()
        item = {"name": m.group(1), "dose": m.group(2), "frequency": m.group(3) or "", "adherence": ""}
        if re.search(r"prescrib|start|give|put you on|advise", ctx):
            d["plan"]["prescriptions"].append(" ".join(x for x in item.values() if x)); add("plan", m.group(0))
        else:
            d["medications"].append(item); add("medications", m.group(0))
    for m in re.finditer(r"allergic to ([a-z ]+?)(?:[.,]|\band\b|$)", low):
        d["allergies"].append(m.group(1).strip()); add("medications", m.group(0).rstrip(" ,.and"))

    m = re.search(r"(\d{2,3})\s*(?:over|/)\s*(\d{2,3})", low)
    if m:
        d["observations"]["vitals"].append(f"BP {m.group(1)}/{m.group(2)} mmHg"); add("observations", m.group(0))
    m = re.search(r"temperature(?: is| of)?\s*([\d.]+)\s*(?:degrees)?\s*(fahrenheit|celsius)?", low)
    if m:
        d["observations"]["vitals"].append(f"Temp {m.group(1)} {(m.group(2) or '')[:1].upper()}".strip()); add("observations", m.group(0))
    m = re.search(r"(?:pulse|heart rate)(?: is| of)?\s*(\d{2,3})", low)
    if m:
        d["observations"]["vitals"].append(f"Pulse {m.group(1)} bpm"); add("observations", m.group(0))

    for m in re.finditer(r"(?:looks like|diagnos(?:is|ed)(?: with)?|impression|consistent with|suspect|likely)\s+([^.]*)", low):
        d["assessment"].append(m.group(1).strip()); add("assessment", m.group(0)); break
    for k in TESTS:
        if k in low:
            d["plan"]["investigations"].append(k); add("plan", k)
    m = re.search(r"(follow[- ]?up[^.]*|come back[^.]*|review (?:in|after)[^.]*)", low)
    if m:
        d["plan"]["follow_up"] = m.group(1).strip(); add("plan", m.group(1))

    bits = []
    p = d["patient"]
    if p["age"] or p["sex"]:
        bits.append(f"{p['age'] + '-year-old ' if p['age'] else ''}{p['sex'].lower() or 'patient'} presents with "
                    f"{d['chief_complaint'] or 'unspecified complaint'}"
                    f"{' for ' + d['hpi']['duration'] if d['hpi']['duration'] else ''}.")
    if d["symptoms"]["negative"]:
        bits.append("Denies " + ", ".join(d["symptoms"]["negative"]) + ".")
    if d["assessment"]:
        bits.append("Assessment: " + "; ".join(d["assessment"]) + ".")
    d["clinical_summary"] = " ".join(bits)
    return d


def analyze(transcript: str):
    t = transcript.strip()
    data, engine, err = None, "heuristic", ""
    if t:
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                data, engine = _claude(t), "claude"
            elif os.getenv("OPENAI_API_KEY"):
                data, engine = _openai(t), "openai"
        except Exception as e:  # fall back, but tell the UI why
            err = f"{type(e).__name__}: {e}"[:200]
            data = None
        if data is None:
            data, engine = _heuristic(t), "heuristic"
    data = _merge(empty(), data or {})
    data["evidence"], total = _ground(data["evidence"], t)
    data["meta"] = {"engine": engine, "evidence_total": total, "evidence_grounded": len(data["evidence"]),
                    "error": err, "words": len(t.split())}
    return data
