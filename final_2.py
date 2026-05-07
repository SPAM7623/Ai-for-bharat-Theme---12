
import os
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from openai import OpenAI

DEBUG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CaseState:

    # =====================================================
    # CORE CASE INFORMATION
    # =====================================================

    intent: Optional[str] = None
    department: Optional[str] = None

    # =====================================================
    # ENTITY STORAGE
    # =====================================================

    entities: Dict[str, Optional[str]] = field(default_factory=dict)

    # metadata tracking
    entity_confidence: Dict[str, float] = field(default_factory=dict)
    entity_source: Dict[str, str] = field(default_factory=dict)

    # =====================================================
    # NLP SIGNALS
    # =====================================================

    sentiment: Optional[str] = None
    confidence: float = 0.85
    language: Optional[str] = None

    # =====================================================
    # USER CONTEXT
    # =====================================================

    current_issue: Optional[str] = None
    multiple_issues: List[str] = field(default_factory=list)

    last_user_message: Optional[str] = None

    # =====================================================
    # FIELD MANAGEMENT
    # =====================================================

    missing_fields: List[str] = field(default_factory=list)

    """
    field_schema format:

    {
        "field_name": {
            "type": "text/boolean/date/location/number",
            "required": True,
            "validation": {},
            "aliases": []
        }
    }
    """

    field_schema: Dict[str, dict] = field(default_factory=dict)

    invalid_fields: List[str] = field(default_factory=list)

    # retry tracking
    field_attempts: Dict[str, int] = field(default_factory=dict)

    max_field_attempts: int = 3

    # =====================================================
    # MASTER FLOW STATE
    # =====================================================

    """
    allowed stages:

    collection
    verification
    correction
    completed
    restart
    handover
    """

    stage: str = "collection"

    # =====================================================
    # CURRENT FIELD CONTEXT
    # =====================================================

    current_field: Optional[str] = None
    last_asked_field: Optional[str] = None

    # =====================================================
    # ACTION TRACKING
    # =====================================================

    """
    examples:

    ask_field
    ask_correction
    ask_field_specific
    verify
    updated
    restart
    handover
    complete
    """

    last_action: Optional[str] = None

    # =====================================================
    # CORRECTION STATE
    # =====================================================

    awaiting_correction: bool = False

    correction_count: int = 0
    last_corrected_field: Optional[str] = None

    field_correction_count: Dict[str, int] = field(default_factory=dict)
    correction_passes: int = 0
    fields_corrected_in_pass: List[str] = field(default_factory=list)
    max_field_corrections: int = 3
    department_field_count: int = 0

    # =====================================================
    # REVERIFICATION STATE (after soft reset)
    # =====================================================

    awaiting_reverify_confirmation: bool = False
    reverify_attempts: int = 0
    max_reverify_attempts: int = 2

    # =====================================================
    # GLOBAL FLOW CONTROL
    # =====================================================

    awaiting_new_issue: bool = False
    just_reset: bool = False

    # =====================================================
    # ATTEMPT CONTROL
    # =====================================================

    attempts: int = 0
    max_attempts: int = 3
    max_correction_count: int = 4

    # =====================================================
    # VERIFICATION / COMPLETION
    # =====================================================

    is_verified: bool = False

    # =====================================================
    # HUMAN ESCALATION
    # =====================================================

    requires_human: bool = False
    escalation_reason: Optional[str] = None

    # =====================================================
    # SESSION TRACKING
    # =====================================================

    conversation_turns: int = 0

    # =====================================================
    # VALIDATION
    # =====================================================

    def __post_init__(self):
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than 0")
        if self.max_field_attempts <= 0:
            raise ValueError("max_field_attempts must be greater than 0")
        if self.max_correction_count <= 0:
            raise ValueError("max_correction_count must be greater than 0")
        if self.max_field_corrections <= 0:
            raise ValueError("max_field_corrections must be greater than 0")

    # =====================================================
    # SAFE ENTITY UPDATE
    # =====================================================

    def update_entity(
        self,
        field_name,
        value,
        confidence=1.0,
        source="system",
        max_length=500
    ):

        if value in [None, ""]:
            return False

        value = str(value).strip()

        if len(value) == 0 or len(value) > max_length:
            return False

        self.entities[field_name] = value
        self.entity_confidence[field_name] = confidence
        self.entity_source[field_name] = source

        return True

    # =====================================================
    # REMOVE ENTITY
    # =====================================================

    def remove_entity(self, field_name):

        self.entities.pop(field_name, None)
        self.entity_confidence.pop(field_name, None)
        self.entity_source.pop(field_name, None)

    # =====================================================
    # FIELD ATTEMPT CONTROL
    # =====================================================

    def increment_field_attempt(self, field_name):

        self.field_attempts[field_name] = (
            self.field_attempts.get(field_name, 0) + 1
        )

        return self.field_attempts[field_name]

    def reset_field_attempt(self, field_name):

        self.field_attempts[field_name] = 0

    def exceeded_field_attempts(self, field_name):

        return (
            self.field_attempts.get(field_name, 0)
            >= self.max_field_attempts
        )

    # =====================================================
    # GLOBAL ATTEMPTS
    # =====================================================

    def increment_attempts(self):

        self.attempts += 1
        return self.attempts

    def reset_attempts(self):

        self.attempts = 0

    # =====================================================
    # CONFIDENCE MANAGEMENT (with attempt decay)
    # =====================================================

    def reduce_confidence(self, amount=0.1):
        """Reduce confidence - penalizes failed extraction"""
        self.confidence = max(0.0, self.confidence - amount)

    def increase_confidence(self, amount=0.05):
        """Increase confidence - rewards successful extraction"""
        self.confidence = min(1.0, self.confidence + amount)

    def apply_attempt_decay(self):
        """Apply decay to confidence based on total attempts (call only on failure)"""
        decay_factor = max(0.6, 1.0 - (self.attempts * 0.08))
        self.confidence = max(0.0, self.confidence * decay_factor)

    # =====================================================
    # CORRECTION TRACKING
    # =====================================================

    def register_correction(self, field_name):

        if self.last_corrected_field == field_name:
            self.correction_count += 1
        else:
            self.correction_count = 1
            self.last_corrected_field = field_name

    def reset_correction_tracking(self):

        self.correction_count = 0
        self.last_corrected_field = None

    # =====================================================
    # SOFT RESET
    # =====================================================

    def soft_reset(self):

        """
        reset conversational flow
        preserve entities + language
        ✓ KEEPS: All extracted data, sentiment, confidence, language
        ✗ CLEARS: attempts, corrections, field attempts, current field

        Used for: Correction loops, high escalation score
        Effect: User restarts but system remembers what they said before
        """

        self.attempts = 0

        self.awaiting_correction = False
        self.awaiting_new_issue = False
        self.awaiting_reverify_confirmation = False

        self.current_field = None
        self.last_asked_field = None

        self.last_action = None

        self.just_reset = True

        self.stage = "collection"

        self.field_attempts = {}
        self.correction_passes = 0
        self.fields_corrected_in_pass = []
        self.reverify_attempts = 0

    # =====================================================
    # CORRECTION TRACKING
    # =====================================================

    def track_field_correction(self, field_name):
        """Track corrections per field and detect drift/repetition"""
        self.field_correction_count[field_name] = (
            self.field_correction_count.get(field_name, 0) + 1
        )

        if field_name not in self.fields_corrected_in_pass:
            self.fields_corrected_in_pass.append(field_name)

        return self.field_correction_count[field_name]

    def detect_correction_drift(self):
        """
        Detect if user is correcting same field too many times
        Returns: (is_drifting, reason)
        """
        if not self.field_correction_count:
            return False, None

        for field_name, count in self.field_correction_count.items():
            if count > self.max_field_corrections:
                return True, f"Field '{field_name}' corrected {count} times"

        return False, None

    def detect_multi_pass_drift(self):
        """
        Detect if user has gone through ALL fields and is correcting again
        Indicates uncertainty or confusion about the data
        """
        if not self.department_field_count or not self.fields_corrected_in_pass:
            return False, None

        unique_fields_corrected = len(self.fields_corrected_in_pass)

        if unique_fields_corrected >= self.department_field_count:
            self.correction_passes += 1

            if self.correction_passes >= 2:
                return True, f"User correcting fields for 2nd time (uncertainty)"

            self.fields_corrected_in_pass = []

        return False, None

    def check_total_corrections_exceed_fields(self):
        """
        If total corrections > number of fields in department,
        user is likely confused or uncertain
        """
        if not self.department_field_count:
            return False, None

        if self.correction_count > self.department_field_count:
            return True, f"Total corrections ({self.correction_count}) exceed fields ({self.department_field_count})"

        return False, None

    # =====================================================
    # LOW CONFIDENCE FIELD DETECTION
    # =====================================================

    def get_low_confidence_fields(self, threshold=0.80):
        """
        After soft reset, identify preserved entities with low confidence
        Returns list of (field_name, confidence, value) tuples
        """
        low_conf_fields = []

        for field_name, value in self.entities.items():
            if not value:
                continue

            confidence = self.entity_confidence.get(field_name, 0.85)

            if confidence < threshold:
                low_conf_fields.append((field_name, confidence, value))

        return low_conf_fields

    # =====================================================
    # FULL RESET
    # =====================================================

    def full_reset(self, preserve_language=True):

        preserved_language = (
            self.language
            if preserve_language
            else None
        )

        self.__dict__.update(
            CaseState().__dict__
        )

        if preserve_language:
            self.language = preserved_language

    # =====================================================
    # HUMAN ESCALATION
    # =====================================================

    def escalate(self, reason):

        self.requires_human = True
        self.escalation_reason = reason

        self.stage = "handover"

    # =====================================================
    # STAGE MANAGEMENT
    # =====================================================

    def set_stage(self, stage_name):

        allowed = [
            "collection",
            "verification",
            "correction",
            "completed",
            "restart",
            "handover"
        ]

        if stage_name in allowed:
            self.stage = stage_name

class InputAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def run(self, audio_path=None, text_input=None):

        # prioritize text input
        if text_input is not None:
            return str(text_input).strip()

        # placeholder for audio transcription
        if audio_path is not None:
            # TODO: integrate ASR here
            return "transcribed text"

        return ""


class UnderstandingAgent:

    def __init__(self):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # =====================================================
    # SAFE JSON PARSER
    # =====================================================

    def safe_json_parse(self, content):

        try:
            return json.loads(content)

        except (json.JSONDecodeError, ValueError, TypeError):

            match = re.search(
                r"\{.*\}",
                content,
                re.DOTALL
            )

            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        return {}

    # =====================================================
    # RISK KEYWORD DETECTION
    # =====================================================

    def detect_risk_indicators(self, text):
        """Detect abuse/danger keywords and escalate sentiment"""
        text_lower = str(text or "").lower()

        critical_keywords = ["abuse", "assault", "violence", "weapon", "gun", "knife"]
        danger_keywords = [
            "threat", "threatened", "threatening", "threaten",
            "attack", "violent",
            "scared", "afraid", "fear", "terrified",
            "hurt", "harm", "injury", "injure"
        ]
        concern_keywords = [
            "hit", "beat", "punch", "kick", "slap",
            "aggressive", "dangerous", "danger"
        ]

        def word_in_text(word):
            return re.search(r'\b' + re.escape(word) + r'\b', text_lower)

        def is_negated(word):
            match = re.search(r'\b(not|no|don\'t|didn\'t|isn\'t|aren\'t|wasn\'t|weren\'t)\s+\w+\s+' + re.escape(word), text_lower)
            return match is not None

        critical_count = sum(1 for kw in critical_keywords if word_in_text(kw) and not is_negated(kw))
        danger_count = sum(1 for kw in danger_keywords if word_in_text(kw) and not is_negated(kw))
        concern_count = sum(1 for kw in concern_keywords if word_in_text(kw) and not is_negated(kw))

        if critical_count >= 1 or danger_count >= 2:
            return "high"

        if danger_count >= 1 or concern_count >= 2:
            return "medium_high"

        if concern_count >= 1:
            return "medium_high"

        return None

    # =====================================================
    # FRUSTRATION & CONFUSION DETECTION
    # =====================================================

    def detect_frustration(self, text, state=None):
        """Detect explicit frustration, confusion, and fatigue signals"""
        text_lower = str(text or "").lower()

        explicit_frustration = [
            "frustrat", "annoyed", "annoying", "annoyed", "tired of",
            "fed up", "enough", "stop", "quit", "give up",
            "waste time", "waste my time", "pointless",
            "ridiculous", "ridiculos", "absurd", "unacceptable",
            "angry", "furious", "enraged", "mad",
            "upset", "unhappy", "dissatisfied",
            "irritated", "irritating",
            "exhausted", "exhausting",
            "sick of", "sick and tired"
        ]

        confusion_phrases = [
            "can't understand", "cannot understand", "don't understand",
            "cannot understand",
            "can't figure", "can't make sense",
            "confusing", "confused", "confuse me",
            "unclear", "not clear", "vague",
            "what does", "what do you mean", "what are you asking",
            "too complicated", "too complex", "too many",
            "complicated", "complex", "bewildering"
        ]

        incompleteness = [
            "don't know", "don't know what", "don't know how",
            "can't", "cannot", "unable to",
            "can't provide", "can't help",
            "not sure", "unsure", "no idea"
        ]

        def find_phrase(phrases):
            return any(re.search(r'\b' + re.escape(phrase) + r'\b', text_lower) for phrase in phrases)

        frustration_count = sum(1 for phrase in explicit_frustration if find_phrase([phrase]))
        confusion_count = sum(1 for phrase in confusion_phrases if find_phrase([phrase]))
        incompleteness_count = sum(1 for phrase in incompleteness if find_phrase([phrase]))

        # Explicit frustration triggers high immediately
        if frustration_count >= 1:
            return "high"

        # Multiple confusion signals trigger high
        if confusion_count >= 2 or incompleteness_count >= 2:
            return "high"

        # Single confusion signal = medium_high
        if confusion_count >= 1 or incompleteness_count >= 1:
            return "medium_high"

        return None

    # =====================================================
    # REPETITION DETECTION
    # =====================================================

    def detect_repetition_fatigue(self, state):
        """Detect if same field has been asked multiple times"""
        if not state.current_field or not state.field_attempts:
            return False

        field_attempt_count = state.field_attempts.get(state.current_field, 0)
        return field_attempt_count >= 3

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize(self, value):

        if not isinstance(value, str):
            return value

        value = value.lower().strip()

        normalization_map = {
            "yeah": "yes",
            "yep": "yes",
            "true": "yes",
            "nope": "no",
            "false": "no"
        }

        return normalization_map.get(
            value,
            value
        )

    # =====================================================
    # FIELD TYPE
    # =====================================================

    def get_field_type(
        self,
        target_field,
        state
    ):

        if not state:
            return "text"

        config = state.field_schema.get(
            target_field,
            {}
        )

        return config.get(
            "type",
            "text"
        )

    # =====================================================
    # FIELD VALIDATION
    # =====================================================

    def validate_field_value(
        self,
        value,
        field_type
    ):

        value = str(value or "").strip()

        if not value:
            return False

        # -------------------------------------------------
        # BOOLEAN
        # -------------------------------------------------

        if field_type == "boolean":

            valid_boolean = [
                "yes",
                "no",
                "true",
                "false",
                "yeah",
                "nope",
                "y",
                "n"
            ]

            return (
                value.lower()
                in valid_boolean
            )

        # -------------------------------------------------
        # DATE / TIME
        # -------------------------------------------------

        if field_type in [
            "date",
            "datetime",
            "date_time"
        ]:

            natural_time_patterns = [
                "today",
                "yesterday",
                "last night",
                "this morning",
                "tonight",
                "few minutes ago",
                "few hours ago"
            ]

            if any(
                pattern in value.lower()
                for pattern in natural_time_patterns
            ):
                return True

            # generic date/number presence
            if re.search(r"\d", value):
                return True

            return False

        # -------------------------------------------------
        # LOCATION
        # -------------------------------------------------

        if field_type == "location":

            return len(value) >= 2

        # -------------------------------------------------
        # NUMBER
        # -------------------------------------------------

        if field_type == "number":

            return bool(
                re.search(r"\d", value)
            )

        # -------------------------------------------------
        # DEFAULT TEXT
        # -------------------------------------------------

        return len(value) >= 2

    # =====================================================
    # FIELD EXTRACTION
    # =====================================================

    def _extract_context_keywords(self, text, target_field):
        """Extract values using field-specific keyword patterns with word boundaries"""
        text_lower = text.lower().strip()
        text_words = text_lower.split()

        # =====================================================
        # SIMPLE "IT'S [VALUE]" OR "ITS [VALUE]" PATTERN
        # =====================================================
        # Handle simple statements like "it's house" or "its house"
        if text_lower.startswith("it's ") or text_lower.startswith("its "):
            # Extract everything after "it's" or "its"
            if text_lower.startswith("it's "):
                value = text[5:].strip()  # Skip "it's "
            else:
                value = text[4:].strip()  # Skip "its "

            # Only return if it's reasonably short (not a full sentence)
            if value and len(value.split()) <= 3:
                return value

        if "date_time" in target_field:
            # Extract date AND time
            date_keywords = ["yesterday", "today", "tomorrow", "tonight", "last night"]
            time_keywords = ["morning", "afternoon", "evening", "night", "noon", "midnight"]

            # Check for explicit date + time combinations
            for date_kw in date_keywords:
                if date_kw in text_lower:
                    for time_kw in time_keywords:
                        if time_kw in text_lower:
                            return f"{date_kw} {time_kw}"
                    return date_kw

            # Check for time alone
            for time_kw in time_keywords:
                if time_kw in text_lower:
                    return time_kw

        elif "date" in target_field:
            date_keywords = [
                "yesterday", "today", "tomorrow",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "last week", "this week", "next week",
                "last month", "this month", "next month",
                "last year", "this year", "next year",
                "tonight", "last night"
            ]

            for kw in date_keywords:
                if kw in text_lower:
                    return kw

        elif "time" in target_field:
            time_keywords = [
                "morning", "afternoon", "evening", "night", "noon", "midnight",
                "early", "late", "dawn", "dusk", "sunset", "sunrise"
            ]

            for kw in time_keywords:
                if kw in text_lower:
                    return kw

        elif "location" in target_field or "place" in target_field or "address" in target_field:
            location_keywords = [
                "house", "home", "office", "street", "park", "restaurant",
                "store", "bank", "hospital", "station", "apartment", "building",
                "school", "college", "university", "mall", "market", "shop",
                "clinic", "pharmacy", "hotel", "cafe", "gym", "library",
                "church", "temple", "mosque", "beach", "mountain", "lake",
                "road", "avenue", "lane", "plaza", "square", "compound"
            ]

            # Try substring matching first (more reliable)
            for kw in location_keywords:
                if kw in text_lower:
                    # Find the word in the text and extract with context
                    kw_idx = text_lower.find(kw)
                    if kw_idx != -1:
                        # Get context around the keyword
                        start = max(0, kw_idx - 30)
                        end = min(len(text), kw_idx + len(kw) + 30)
                        context = text[start:end].strip()
                        return context

        return None

    def extract_field(
        self,
        text,
        target_field,
        state=None
    ):

        field_type = self.get_field_type(
            target_field,
            state
        )

        keyword_match = self._extract_context_keywords(text, target_field)
        if keyword_match:
            return {target_field: keyword_match}

        # =================================================
        # CHECK FOR "I DON'T KNOW" / "CAN'T HELP" PHRASES
        # =================================================
        # Treat as valid answer that doesn't fail extraction

        text_lower = str(text or "").lower().strip()
        unknown_phrases = [
            "i don't know",
            "i dont know",
            "don't know",
            "dont know",
            "no idea",
            "not sure",
            "unsure",
            "i don't understand",
            "i dont understand",
            "don't understand",
            "dont understand",
            "i don't get it",
            "i dont get it",
            "i can't help",
            "i cant help",
            "can't help",
            "cant help",
            "not available",
            "unknown"
        ]

        if any(phrase in text_lower for phrase in unknown_phrases):
            return {target_field: "unknown"}

        # =================================================
        # DETERMINISTIC BOOLEAN EXTRACTION
        # =================================================

        if field_type == "boolean":

            normalized = self.normalize(text)

            positive = [
                "yes",
                "yeah",
                "y",
                "true"
            ]

            negative = [
                "no",
                "nope",
                "n",
                "false"
            ]

            if normalized in positive:

                return {
                    target_field: "yes"
                }

            if normalized in negative:

                return {
                    target_field: "no"
                }

            return {}

        # =================================================
        # LLM FIELD EXTRACTION
        # =================================================

        # Special instructions for date/time fields
        date_time_instructions = ""
        if "date" in target_field.lower() or "time" in target_field.lower():
            date_time_instructions = """
SPECIAL INSTRUCTIONS FOR DATE/TIME:
- Extract relative dates: yesterday, today, tomorrow, last week, next month, etc.
- Extract days: monday, tuesday, wednesday, thursday, friday, saturday, sunday
- Extract times: morning, afternoon, evening, night, noon, midnight, dawn, dusk
- Preserve exact phrasing (e.g., "last night", "this afternoon", "next week")
- Examples:
  * "it happened yesterday" → "yesterday"
  * "yesterday evening" → "yesterday evening"
  * "last week friday" → "last week friday"
  * "this morning at 9am" → "this morning"
- Do NOT try to convert to dates, just preserve the natural language
"""

        prompt = f"""
You are extracting a structured field value.

Target Field:
{target_field}

Field Type:
{field_type}

User Input:
"{text}"

Rules:
- Return ONLY valid JSON
- Extract only if clearly relevant
- Do NOT hallucinate
- Do NOT invent values
- Preserve natural language time expressions
- If unclear return empty JSON
{date_time_instructions}

Format:
{{"{target_field}": "value"}}

OR

{{}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        if not parsed:
            return {}

        cleaned = {}

        for key, value in parsed.items():

            normalized = self.normalize(value)

            valid = self.validate_field_value(
                normalized,
                field_type
            )

            if not valid:
                continue

            cleaned[key] = normalized

        return cleaned

    # =====================================================
    # CORRECTION EXTRACTION
    # =====================================================

    def extract_correction(
        self,
        text,
        state=None
    ):

        current_entities = (
            state.entities
            if state else {}
        )

        prompt = f"""
You are correcting structured complaint data.

Current Data:
{json.dumps(current_entities, indent=2)}

User Input:
"{text}"

Rules:
- Extract ONLY corrected fields
- Do NOT return unchanged fields
- Do NOT hallucinate
- If user says:
  "change X to Y"
  return updated value
- If unclear return empty JSON

Return ONLY valid JSON.

Example:
{{"field_name": "updated_value"}}

OR

{{}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        cleaned = {}

        for key, value in parsed.items():

            if value in [None, ""]:
                continue

            cleaned[key] = self.normalize(value)

        return cleaned

    # =====================================================
    # FULL EXTRACTION
    # =====================================================

    def extract_full(
        self,
        text
    ):

        prompt = f"""
You are an intelligent extraction system
for a public helpline.

Return ONLY valid JSON.

Format:
{{
    "intent": "...",
    "entities": {{}},
    "sentiment": "low/medium/high",
    "confidence": number,
    "language": "english/hindi/tamil/mixed"
}}

Rules:
- Extract meaningful structured entities
- Keep values short and normalized
- Do NOT hallucinate
- Do NOT guess missing fields

Input:
{text}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        cleaned = {}

        for key, value in parsed.items():

            # entities block
            if isinstance(value, dict):

                cleaned[key] = {}

                for ek, ev in value.items():

                    if ev in [None, ""]:
                        continue

                    cleaned[key][ek] = (
                        self.normalize(ev)
                    )

            else:

                cleaned[key] = (
                    self.normalize(value)
                )

        return cleaned

    # =====================================================
    # MAIN EXTRACTION ROUTER
    # =====================================================

    def extract(
        self,
        text,
        mode="full",
        target_field=None,
        state=None
    ):

        text = str(text or "").strip()

        if not text:
            return {}

        # -------------------------------------------------
        # FULL EXTRACTION
        # -------------------------------------------------

        if mode == "full":

            return self.extract_full(text)

        # -------------------------------------------------
        # FIELD EXTRACTION
        # -------------------------------------------------

        if mode == "field":

            return self.extract_field(
                text=text,
                target_field=target_field,
                state=state
            )

        # -------------------------------------------------
        # CORRECTION EXTRACTION
        # -------------------------------------------------

        if mode == "correction":

            return self.extract_correction(
                text=text,
                state=state
            )

        return {}

    # =====================================================
    # STATE UPDATE
    # =====================================================

    def update_state(
        self,
        state,
        result
    ):

        if not result:
            return state

        # -------------------------------------------------
        # FIELD UPDATE
        # -------------------------------------------------

        if (
            isinstance(result, dict)
            and "entities" not in result
        ):

            for field, value in result.items():

                state.update_entity(
                    field_name=field,
                    value=value,
                    confidence=0.9,
                    source="understanding_agent"
                )

            return state

        # -------------------------------------------------
        # FULL EXTRACTION UPDATE
        # -------------------------------------------------

        if result.get("language"):

            if not state.language:
                state.language = (
                    result.get("language")
                )

        if result.get("intent"):

            state.intent = (
                result.get("intent")
            )

        if result.get("sentiment"):

            state.sentiment = (
                result.get("sentiment")
            )

        risk_sentiment = self.detect_risk_indicators(state.last_user_message)
        if risk_sentiment:
            if state.sentiment != "high":
                state.sentiment = risk_sentiment

        frustration_sentiment = self.detect_frustration(state.last_user_message, state)
        if frustration_sentiment:
            if state.sentiment != "high":
                state.sentiment = frustration_sentiment

        if result.get("confidence") is not None:

            state.confidence = (
                result.get("confidence", 0.0)
            )

        entities = result.get(
            "entities",
            {}
        )

        for field, value in entities.items():

            state.update_entity(
                field_name=field,
                value=value,
                confidence=result.get(
                    "confidence",
                    0.8
                ),
                source="full_extraction"
            )

        return state


class CaseBuilderAgent:

    def __init__(
        self,
        departments
    ):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.departments = departments

    # =====================================================
    # SAFE JSON PARSER
    # =====================================================

    def safe_json_parse(self, content):

        try:
            return json.loads(content)

        except (json.JSONDecodeError, ValueError, TypeError):

            match = re.search(
                r"\{.*\}",
                content,
                re.DOTALL
            )

            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        return {}

    # =====================================================
    # VALUE FORMATTING & FALLBACK
    # =====================================================

    def format_extracted_value(self, value, field_name):
        """Compact and clean extracted values - always returns a value"""
        if not value:
            return None

        value_str = str(value).strip()

        if not value_str:
            return None

        if "accused" in field_name.lower() or "suspect" in field_name.lower():
            if any(word in value_str.lower() for word in ["i don't know", "unknown", "no one", "nobody"]):
                return "Unknown perpetrator"
            return value_str[:100] if len(value_str) > 100 else value_str

        if "date" in field_name.lower() or "time" in field_name.lower():
            return value_str.lower()

        if "location" in field_name.lower() or "place" in field_name.lower():
            return value_str

        if "danger" in field_name.lower() or "risk" in field_name.lower():
            normalized = value_str.lower()
            if normalized in ["yes", "true", "y"]:
                return "yes"
            if normalized in ["no", "false", "n"]:
                return "no"
            return value_str

        return value_str[:200] if len(value_str) > 200 else value_str

    def check_fallback_acceptable(self, user_text, field_name):
        """Check if user explicitly said value is unknown/not applicable"""
        text_lower = user_text.lower().strip()

        not_applicable = [
            "n/a", "not applicable", "doesn't apply",
            "not relevant", "skip this", "doesn't matter"
        ]

        not_known = [
            "i don't know", "unknown", "don't know",
            "no clue", "no idea", "can't say", "not sure"
        ]

        if any(phrase in text_lower for phrase in not_applicable):
            return "not_applicable"

        if any(phrase in text_lower for phrase in not_known):
            return "unknown"

        return None

    # =====================================================
    # DEPARTMENT LOOKUP
    # =====================================================

    def get_department_by_name(
        self,
        department_name
    ):

        for department in self.departments:

            if (
                department["name"]
                == department_name
            ):
                return department

        return None

    # =====================================================
    # DEPARTMENT LOOKUP BY CODE
    # =====================================================

    def get_department_by_code(
        self,
        code
    ):

        for department in self.departments:

            if department["code"] == code:
                return department

        return None

    # =====================================================
    # DEFAULT FIELD TYPE INFERENCE
    # =====================================================

    def infer_field_type(
        self,
        field_name
    ):

        field_name = (
            str(field_name)
            .lower()
            .strip()
        )

        # -------------------------------------------------
        # BOOLEAN
        # -------------------------------------------------

        boolean_patterns = [
            "danger",
            "repeat",
            "urgent",
            "emergency",
            "violence",
            "injured",
            "armed",
            "threat"
        ]

        if any(
            pattern in field_name
            for pattern in boolean_patterns
        ):
            return "boolean"

        # -------------------------------------------------
        # DATE / TIME
        # -------------------------------------------------

        datetime_patterns = [
            "date",
            "time",
            "incident_time",
            "incident_date",
            "datetime"
        ]

        if any(
            pattern in field_name
            for pattern in datetime_patterns
        ):
            return "datetime"

        # -------------------------------------------------
        # LOCATION
        # -------------------------------------------------

        location_patterns = [
            "location",
            "address",
            "place",
            "area",
            "landmark"
        ]

        if any(
            pattern in field_name
            for pattern in location_patterns
        ):
            return "location"

        # -------------------------------------------------
        # NUMBER
        # -------------------------------------------------

        number_patterns = [
            "phone",
            "mobile",
            "amount",
            "number",
            "otp",
            "pin",
            "transaction"
        ]

        if any(
            pattern in field_name
            for pattern in number_patterns
        ):
            return "number"

        # -------------------------------------------------
        # DEFAULT
        # -------------------------------------------------

        return "text"

    # =====================================================
    # DEPARTMENT MAPPING
    # =====================================================

    def map_department(
        self,
        intent,
        entities=None,
        text=None
    ):

        prompt = f"""
You are a strict routing engine
for a public helpline.

Return ONLY valid JSON.

Format:
{{
    "code": "DEPARTMENT_CODE",
    "confidence": 0.0
}}

Departments:
{json.dumps(self.departments, indent=2)}

Intent:
{intent}

Entities:
{json.dumps(entities or {}, indent=2)}

User Input:
{text}

IMPORTANT:
- Physical theft/loss belongs to Police
- Online fraud belongs to Cyber Crime
- Do NOT confuse physical item loss
  with digital fraud
- Do NOT guess blindly

Return ONLY JSON.
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        department = self.get_department_by_code(
            parsed.get("code")
        )

        if not department:
            return None

        return department

    # =====================================================
    # SAFE ENTITY ALIGNMENT
    # =====================================================

    def align_entities(
        self,
        entities,
        field_schema,
        intent
    ):

        prompt = f"""
Map extracted entities
to department fields.

Intent:
{intent}

Entities:
{json.dumps(entities, indent=2)}

Field Schema:
{json.dumps(field_schema, indent=2)}

Rules:
- Map ONLY clearly matching fields
- NEVER fabricate values
- NEVER force missing fields
- NEVER duplicate unrelated values
- If uncertain do NOT map
- Return ONLY valid JSON

Example:
{{
    "field_name": "value"
}}

OR

{{}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        aligned = {}

        for field, value in parsed.items():

            if field not in field_schema:
                continue

            if value in [None, ""]:
                continue

            aligned[field] = value

        return aligned

    # =====================================================
    # BUILD FIELD SCHEMA
    # =====================================================

    def build_field_schema(
        self,
        department
    ):

        schema = {}

        required_fields = department.get(
            "required_fields",
            []
        )

        for field in required_fields:

            # -------------------------------------------------
            # STRUCTURED FIELD CONFIG
            # -------------------------------------------------

            if isinstance(field, dict):

                field_name = field["name"]

                schema[field_name] = {
                    "type": field.get(
                        "type",
                        self.infer_field_type(
                            field_name
                        )
                    ),
                    "required": field.get(
                        "required",
                        True
                    ),
                    "validation": field.get(
                        "validation",
                        {}
                    ),
                    "aliases": field.get(
                        "aliases",
                        []
                    )
                }

            # -------------------------------------------------
            # SIMPLE STRING FIELD
            # -------------------------------------------------

            else:

                inferred_type = (
                    self.infer_field_type(
                        field
                    )
                )

                schema[field] = {
                    "type": inferred_type,
                    "required": True,
                    "validation": {},
                    "aliases": []
                }

        return schema

    # =====================================================
    # ENSURE ENTITY KEYS
    # =====================================================

    def ensure_entity_keys(
        self,
        state,
        field_schema
    ):

        for field_name in field_schema:

            if (
                field_name
                not in state.entities
            ):

                state.entities[field_name] = None

    # =====================================================
    # COMPUTE MISSING FIELDS
    # =====================================================

    def compute_missing_fields(
        self,
        state,
        field_schema
    ):

        missing = []

        for field_name, config in field_schema.items():

            if not config.get(
                "required",
                True
            ):
                continue

            value = state.entities.get(
                field_name
            )

            if value in [None, ""]:
                missing.append(field_name)

        return missing

    # =====================================================
    # MAIN CASE UPDATE
    # =====================================================

    def update_case(
        self,
        state,
        extracted,
        mode="normal"
    ):

        # -------------------------------------------------
        # PREVENT MODIFICATION AFTER COMPLETION
        # -------------------------------------------------

        if (
            state.is_verified
            and mode == "normal"
        ):
            return state

        # =================================================
        # GLOBAL UPDATE
        # =================================================

        if mode == "global":

            if extracted.get("intent"):
                state.intent = (
                    extracted.get("intent")
                )

            if extracted.get("sentiment"):
                state.sentiment = (
                    extracted.get("sentiment")
                )

            if (
                extracted.get("confidence")
                is not None
            ):

                state.confidence = (
                    extracted.get(
                        "confidence",
                        0.0
                    )
                )

            if extracted.get("language"):

                state.language = (
                    extracted.get("language")
                )

            for field, value in extracted.get(
                "entities",
                {}
            ).items():

                formatted_value = self.format_extracted_value(value, field)
                if formatted_value:
                    state.update_entity(
                        field_name=field,
                        value=formatted_value,
                        confidence=extracted.get(
                            "confidence",
                            0.85
                        ),
                        source="global_extraction"
                    )

        # =================================================
        # CORRECTION UPDATE
        # =================================================

        elif mode == "correction":

            for field, value in extracted.items():

                if field.startswith("__"):
                    continue

                formatted_value = self.format_extracted_value(value, field)
                if formatted_value:
                    updated = state.update_entity(
                        field_name=field,
                        value=formatted_value,
                        confidence=0.95,
                        source="correction"
                    )

                    if updated:
                        state.register_correction(
                            field
                        )
                        state.track_field_correction(
                            field
                        )

            state.reduce_confidence(0.05)

        # =================================================
        # REVERIFY UPDATE (after soft reset)
        # Does NOT inflate correction metrics
        # =================================================

        elif mode == "reverify":

            for field, value in extracted.items():

                if field.startswith("__"):
                    continue

                formatted_value = self.format_extracted_value(value, field)
                if formatted_value:
                    state.update_entity(
                        field_name=field,
                        value=formatted_value,
                        confidence=0.95,
                        source="reverify"
                    )

        # =================================================
        # NORMAL UPDATE
        # =================================================

        else:

            for field, value in extracted.items():

                if field.startswith("__"):
                    continue

                formatted_value = self.format_extracted_value(value, field)
                if formatted_value:
                    state.update_entity(
                        field_name=field,
                        value=formatted_value,
                        confidence=0.9,
                        source="normal_update"
                    )

            state.increase_confidence(0.05)

            state.reset_correction_tracking()

        # =================================================
        # DEPARTMENT ASSIGNMENT
        # =================================================

        department = None

        # only assign once
        if not state.department:

            department = self.map_department(
                intent=state.intent,
                entities=state.entities,
                text=state.current_issue
            )

            if department:
                state.department = (
                    department["name"]
                )
                state.department_field_count = len(
                    department.get("required_fields", [])
                )

        else:

            department = (
                self.get_department_by_name(
                    state.department
                )
            )
            if department:
                state.department_field_count = len(
                    department.get("required_fields", [])
                )

        # -------------------------------------------------
        # NO DEPARTMENT FOUND
        # -------------------------------------------------

        if not department:
            return state

        # =================================================
        # FIELD SCHEMA
        # =================================================

        field_schema = self.build_field_schema(
            department
        )

        state.field_schema = field_schema

        # =================================================
        # SAFE ALIGNMENT
        # =================================================

        aligned_entities = self.align_entities(
            entities=state.entities,
            field_schema=field_schema,
            intent=state.intent
        )

        for field, value in aligned_entities.items():

            # prevent overwrite
            if state.entities.get(field):
                continue

            state.update_entity(
                field_name=field,
                value=value,
                confidence=0.8,
                source="alignment"
            )

        # =================================================
        # ENSURE ENTITY KEYS
        # =================================================

        self.ensure_entity_keys(
            state,
            field_schema
        )

        # =================================================
        # COMPUTE MISSING FIELDS
        # =================================================

        state.missing_fields = (
            self.compute_missing_fields(
                state,
                field_schema
            )
        )

        return state

from openai import OpenAI


class InteractionAgent:

    def __init__(self):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # =====================================================
    # CORE RESPONSE GENERATOR
    # =====================================================

    def gen(self, state, instruction):

        prompt = f"""
You are a polite and professional
public helpline assistant.

Respond in:
{state.language or "English"}

Rules:
- Be natural
- Be short
- Be conversational
- Be emotionally appropriate
- Do NOT sound robotic
- Ask only ONE thing at a time

Instruction:
{instruction}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    # =====================================================
    # FIELD HUMANIZER
    # =====================================================

    def humanize(self, field_name):

        return (
            field_name
            .replace("_", " ")
            .strip()
        )

    # =====================================================
    # FIELD CONFIG
    # =====================================================

    def get_field_config(
        self,
        state,
        field_name
    ):

        return state.field_schema.get(
            field_name,
            {}
        )

    # =====================================================
    # FIELD TYPE
    # =====================================================

    def get_field_type(
        self,
        state,
        field_name
    ):

        config = self.get_field_config(
            state,
            field_name
        )

        return config.get(
            "type",
            "text"
        )

    # =====================================================
    # ESCALATION CHECK
    # =====================================================

    def check_escalation(self, state):

        # -------------------------------------------------
        # HUMAN TAKEOVER
        # -------------------------------------------------

        if state.requires_human:
            return "handover"

        # repeated failures + high sentiment
        if (
            state.sentiment == "high"
            and state.attempts >= state.max_attempts
        ):
            return "handover"

        # too many corrections
        if state.correction_count >= 4:
            return "handover"

        # -------------------------------------------------
        # GLOBAL RESET
        # -------------------------------------------------

        if (
            state.awaiting_correction
            and state.attempts >= 2
        ):
            return "reset"

        return None

    # =====================================================
    # ASK MISSING FIELD
    # =====================================================

    def ask_missing(self, state):

        escalation = self.check_escalation(
            state
        )

        if escalation == "reset":
            return self.ask_rephrase(state)

        if escalation == "handover":
            return self.human_handover(state)

        # -------------------------------------------------
        # NO MISSING FIELDS
        # -------------------------------------------------

        if not state.missing_fields:

            state.last_action = "clarify"

            return {
                "text": self.gen(
                    state,
                    "Ask the user to clarify the issue."
                ),
                "action": "clarify"
            }

        # -------------------------------------------------
        # CURRENT FIELD
        # -------------------------------------------------

        field_name = state.missing_fields[0]

        state.current_field = field_name
        state.last_asked_field = field_name

        state.last_action = "ask_field"

        field_type = self.get_field_type(
            state,
            field_name
        )

        nice_field = self.humanize(
            field_name
        )

        # -------------------------------------------------
        # BOOLEAN FIELD
        # -------------------------------------------------

        if field_type == "boolean":

            instruction = f"""
Ask clearly for:
{nice_field}

Expect only yes or no.
"""

        # -------------------------------------------------
        # GENERAL FIELD
        # -------------------------------------------------

        else:

            instruction = f"""
Ask the user to provide:
{nice_field}
"""

        return {
            "text": self.gen(
                state,
                instruction
            ),
            "action": "ask_field",
            "field": field_name
        }

    # =====================================================
    # RE-VERIFY LOW CONFIDENCE FIELDS (after soft reset)
    # =====================================================

    def ask_reverify_low_confidence(self, state, confidence_threshold=0.80):

        low_conf_fields = state.get_low_confidence_fields(threshold=confidence_threshold)

        if not low_conf_fields:
            return None

        state.last_action = "reverify_low_confidence"
        state.awaiting_reverify_confirmation = True
        state.reverify_attempts = 0

        field_name, confidence, value = low_conf_fields[0]

        state.current_field = field_name
        state.last_asked_field = field_name

        nice_field = self.humanize(field_name)

        instruction = f"""
Before we continue, I want to double-check a detail.

You mentioned {nice_field} as: {value}

Is that correct?
"""

        return {
            "text": self.gen(
                state,
                instruction
            ),
            "action": "reverify_field",
            "field": field_name
        }

    # =====================================================
    # ASK FOR CORRECTED VALUE DURING REVERIFY
    # =====================================================

    def ask_reverify_correction(self, state):

        field_name = state.current_field
        nice_field = self.humanize(field_name)

        instruction = f"""
What is the correct value for {nice_field}?
"""

        state.last_action = "reverify_correction"
        state.reverify_attempts += 1

        return {
            "text": self.gen(
                state,
                instruction
            ),
            "action": "reverify_correction",
            "field": field_name
        }

    # =====================================================
    # ASK CORRECTION OR RESTART
    # CASE A
    # =====================================================

    def ask_correction_or_restart(self, state):

        escalation = self.check_escalation(
            state
        )

        if escalation == "reset":
            return self.ask_rephrase(state)

        if escalation == "handover":
            return self.human_handover(state)

        state.set_stage("correction")

        state.last_action = (
            "ask_correction_or_restart"
        )

        return {
            "text": self.gen(
                state,
                """
Ask the user whether
they want to correct
something or restart
the complaint.
"""
            ),
            "action": "ask_correction_or_restart"
        }

    # =====================================================
    # ASK WHAT TO CORRECT
    # CASE B1
    # =====================================================

    def ask_what_to_correct(self, state):

        escalation = self.check_escalation(
            state
        )

        if escalation == "reset":
            return self.ask_rephrase(state)

        if escalation == "handover":
            return self.human_handover(state)

        state.awaiting_correction = True

        state.set_stage("correction")

        state.last_action = (
            "ask_what_to_correct"
        )

        return {
            "text": self.gen(
                state,
                """
Ask the user what exactly
they want to correct.
"""
            ),
            "action": "ask_what_to_correct"
        }

    # =====================================================
    # FIELD-SPECIFIC CORRECTION
    # CASE B2
    # =====================================================

    def ask_field_specific(
        self,
        state,
        field_name
    ):

        escalation = self.check_escalation(
            state
        )

        if escalation == "reset":
            return self.ask_rephrase(state)

        if escalation == "handover":
            return self.human_handover(state)

        state.awaiting_correction = True

        state.current_field = field_name
        state.last_asked_field = field_name

        state.set_stage("correction")

        state.last_action = (
            "ask_field_specific"
        )

        field_type = self.get_field_type(
            state,
            field_name
        )

        nice_field = self.humanize(
            field_name
        )

        # -------------------------------------------------
        # BOOLEAN FIELD
        # -------------------------------------------------

        if field_type == "boolean":

            instruction = f"""
Ask the user for the
correct value of:
{nice_field}

Expect yes or no.
"""

        # -------------------------------------------------
        # GENERAL FIELD
        # -------------------------------------------------

        else:

            instruction = f"""
Tell the user you understood
that {nice_field} is incorrect.

Then ask for the correct value.
"""

        return {
            "text": self.gen(
                state,
                instruction
            ),
            "action": "ask_field_specific",
            "field": field_name
        }

    # =====================================================
    # CORRECTION APPLIED
    # =====================================================

    def correction_applied(self, state):

        state.last_action = "updated"

        state.set_stage("verification")

        return {
            "text": self.gen(
                state,
                """
Tell the user the
information has been updated.
"""
            ),
            "action": "updated"
        }

    # =====================================================
    # ASK MORE CORRECTIONS
    # =====================================================

    def ask_more_corrections(self, state):

        escalation = self.check_escalation(
            state
        )

        if escalation == "reset":
            return self.ask_rephrase(state)

        if escalation == "handover":
            return self.human_handover(state)

        state.last_action = (
            "ask_more_corrections"
        )

        state.set_stage("correction")

        return {
            "text": self.gen(
                state,
                """
Ask the user whether
they want to correct
anything else.
"""
            ),
            "action": "ask_more_corrections"
        }

    # =====================================================
    # GLOBAL RESET
    # =====================================================

    def ask_rephrase(self, state):

        state.soft_reset()

        state.set_stage("restart")

        state.last_action = "restart"

        return {
            "text": self.gen(
                state,
                """
Ask the user to explain
the complaint again
from the beginning.
"""
            ),
            "action": "restart"
        }

    # =====================================================
    # HUMAN HANDOVER
    # =====================================================

    def human_handover(self, state):

        state.escalate(
            "conversation_complexity"
        )

        state.last_action = "handover"

        return {
            "text": self.gen(
                state,
                """
Inform the user that
they are being connected
to a human agent for
further assistance.
"""
            ),
            "action": "handover"
        }


class VerificationAgent:

    def __init__(self):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # =====================================================
    # SUMMARY GENERATION
    # =====================================================

    def summary(
        self,
        state,
        field_schema
    ):

        structured_data = {}

        for field_name in field_schema:

            structured_data[field_name] = (
                state.entities.get(field_name)
            )

        prompt = f"""
Create a clean verification summary
for a public helpline complaint.

Complaint Data:
{json.dumps(structured_data, indent=2)}

Rules:
- Include ALL available fields
- Keep it short
- Keep it natural
- Do NOT hallucinate
- Do NOT omit fields
- End by asking:
  "Is this correct?"

Return text only.
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize(self, text):

        return (
            str(text or "")
            .lower()
            .strip()
        )

    # =====================================================
    # PURE YES DETECTION
    # =====================================================

    def is_yes(self, text):

        t = self.normalize(text)

        positive = [
            "yes",
            "yeah",
            "y",
            "correct",
            "right",
            "looks good",
            "perfect"
        ]

        return t in positive

    # =====================================================
    # PURE REJECTION
    # CASE A
    # =====================================================

    def is_pure_rejection(self, text):

        t = self.normalize(text)

        negative = [
            "no",
            "nope",
            "wrong",
            "incorrect",
            "not correct"
        ]

        return t in negative

    # =====================================================
    # EXPLICIT CORRECTION
    # CASE C + D
    # correction overrides rejection
    # =====================================================

    def is_explicit_correction(self, text):

        t = self.normalize(text)

        patterns = [
            r"(change|replace|correct|update|set|make).*(to|with)",
            r"\bto\b",
            r"\bshould be\b",
            r"\bis\b.*\bchange\b"
        ]

        return any(
            re.search(pattern, t)
            for pattern in patterns
        )

    # =====================================================
    # VAGUE CORRECTION
    # CASE B1
    # =====================================================

    def is_vague_correction(self, text):

        t = self.normalize(text)

        vague_patterns = [
            "this is wrong",
            "something is wrong",
            "wrong details",
            "not right",
            "incorrect information"
        ]

        return any(
            pattern in t
            for pattern in vague_patterns
        )

    # =====================================================
    # FIELD TARGET HINT
    # CASE B2
    # =====================================================

    def detect_field_hint(
        self,
        text,
        state
    ):

        t = self.normalize(text)

        # explicit correction overrides
        if self.is_explicit_correction(t):
            return None

        for field_name, value in state.entities.items():

            if not value:
                continue

            value_text = (
                str(value)
                .lower()
                .strip()
            )

            if (
                value_text in t
                and "wrong" in t
            ):
                return field_name

        return None

    # =====================================================
    # MAIN CLASSIFIER
    # =====================================================

    def classify(
        self,
        text,
        state=None
    ):

        t = self.normalize(text)

        # -------------------------------------------------
        # CASE C + D
        # explicit correction overrides rejection
        # -------------------------------------------------

        if self.is_explicit_correction(t):

            return {
                "intent": "correction",
                "action": "apply_correction"
            }

        # -------------------------------------------------
        # CASE A
        # pure rejection
        # -------------------------------------------------

        if self.is_pure_rejection(t):

            return {
                "intent": "reject",
                "action": "ask_correction_or_restart"
            }

        # -------------------------------------------------
        # CASE B1
        # vague correction
        # -------------------------------------------------

        if self.is_vague_correction(t):

            return {
                "intent": "vague_correction",
                "action": "ask_what_to_correct"
            }

        # -------------------------------------------------
        # CASE B2
        # field-specific correction
        # -------------------------------------------------

        if state:

            hinted_field = self.detect_field_hint(
                t,
                state
            )

            if hinted_field:

                return {
                    "intent": "field_hint",
                    "action": "ask_field_specific",
                    "field": hinted_field
                }

        # -------------------------------------------------
        # PURE CONFIRMATION
        # -------------------------------------------------

        if self.is_yes(t):

            return {
                "intent": "confirm",
                "action": "complete"
            }

        # -------------------------------------------------
        # DEFAULT SAFE FALLBACK
        # -------------------------------------------------

        return {
            "intent": "unclear",
            "action": "clarify"
        }

DEPARTMENTS = [
    {
        "name": "Police",
        "code": "POL",
        "required_fields": [
            "incident_type",
            "date_time_of_incident",
            "accused_details",
            "immediate_danger",
            "location_of_incident"

        ]
    },
    {
        "name": "Women Safety",
        "code": "WOM",
        "required_fields": [
            "harassment_type",
            "location_of_incident",
            "suspect_details",
            "repeat_offense"
        ]
    },
    {
        "name": "Traffic Police",
        "code": "TRAFFIC",
        "required_fields": [
            "vehicle_number",
            "violation_type",
            "location",
            "time"
        ]
    },
    {
        "name": "Cyber Crime",
        "code": "CYBER",
        "required_fields": [
            "fraud_type",
            "platform",
            "transaction_details"
        ]
    }
]


class DecisionAgent:

    def __init__(self):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # =====================================================
    # SAFE JSON PARSER
    # =====================================================

    def safe_json_parse(self, content):

        try:
            return json.loads(content)

        except (json.JSONDecodeError, ValueError, TypeError):

            match = re.search(
                r"\{.*\}",
                content,
                re.DOTALL
            )

            if match:
                try:
                    return json.loads(match.group())
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        return {}

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize(self, text):

        return (
            str(text or "")
            .lower()
            .strip()
        )

    # =====================================================
    # PURE YES
    # =====================================================

    def is_pure_yes(self, text):

        positive = [
            "yes",
            "yeah",
            "y",
            "correct",
            "right",
            "looks good",
            "perfect"
        ]

        return text in positive

    # =====================================================
    # PURE REJECTION
    # CASE A
    # =====================================================

    def is_pure_rejection(self, text):

        negative = [
            "no",
            "nope",
            "wrong",
            "incorrect",
            "not correct"
        ]

        return text in negative

    # =====================================================
    # EXPLICIT CORRECTION
    # CASE C + D
    # correction overrides rejection
    # =====================================================

    def detect_explicit_correction(self, text):

        patterns = [
            r"(change|replace|correct|update|set|make).*(to|with)",
            r"\bshould be\b",
            r"\bis\b.*\bchange\b",
            r"\bchange\b.*\binto\b"
        ]

        return any(
            re.search(pattern, text)
            for pattern in patterns
        )

    # =====================================================
    # VAGUE CORRECTION
    # CASE B1
    # =====================================================

    def detect_vague_correction(self, text):

        vague_patterns = [
            "this is wrong",
            "something is wrong",
            "wrong details",
            "incorrect information",
            "not right"
        ]

        return any(
            pattern in text
            for pattern in vague_patterns
        )

    # =====================================================
    # FIELD TARGET HINT
    # CASE B2
    # =====================================================

    def detect_field_hint(
        self,
        text,
        state
    ):

        if state.awaiting_correction:
            return None

        for field_name, value in state.entities.items():

            if not value:
                continue

            entity_value = (
                str(value)
                .lower()
                .strip()
            )

            if (
                entity_value in text
                and "wrong" in text
            ):
                return field_name

        return None

    # =====================================================
    # FIELD RESPONSE
    # =====================================================

    def is_field_response(self, state):

        return (
            state.last_action == "ask_field"
            and state.last_asked_field is not None
        )

    # =====================================================
    # CORRECTION RESPONSE
    # =====================================================

    def is_correction_response(self, state):

        return (
            state.awaiting_correction
            and state.last_asked_field is not None
        )

    # =====================================================
    # MAIN DECISION FUNCTION
    # =====================================================

    def decide(
        self,
        state,
        user_input
    ):

        text = self.normalize(user_input)

        # =================================================
        # PRIORITY 1
        # EXPLICIT CORRECTION
        # CASE C + D
        # correction overrides rejection
        # =================================================

        if self.detect_explicit_correction(text):

            return {
                "intent": "provide_correction",
                "action": "apply_correction",
                "reason": "explicit_correction",
                "confidence_score": 0.98
            }

        # =================================================
        # PRIORITY 2
        # FIELD RESPONSE
        # =================================================

        if self.is_field_response(state):

            return {
                "intent": "provide_field",
                "action": "fill_field",
                "reason": "field_response",
                "confidence_score": 0.95
            }

        # =================================================
        # PRIORITY 3
        # CORRECTION RESPONSE
        # =================================================

        if self.is_correction_response(state):

            return {
                "intent": "provide_correction",
                "action": "apply_correction",
                "reason": "correction_response",
                "confidence_score": 0.95
            }

        # =================================================
        # PRIORITY 4
        # PURE REJECTION
        # CASE A
        # =================================================

        if self.is_pure_rejection(text):

            return {
                "intent": "reject",
                "action": "ask_correction_or_restart",
                "reason": "pure_rejection",
                "confidence_score": 0.9
            }

        # =================================================
        # PRIORITY 5
        # VAGUE CORRECTION
        # CASE B1
        # =================================================

        if self.detect_vague_correction(text):

            return {
                "intent": "vague_correction",
                "action": "ask_what_to_correct",
                "reason": "vague_correction",
                "confidence_score": 0.9
            }

        # =================================================
        # PRIORITY 6
        # FIELD TARGET HINT
        # CASE B2
        # =================================================

        hinted_field = self.detect_field_hint(
            text,
            state
        )

        if (
            hinted_field
            and not state.missing_fields
        ):

            return {
                "intent": "field_hint",
                "action": "ask_field_specific",
                "field": hinted_field,
                "reason": "field_target_hint",
                "confidence_score": 0.88
            }

        # =================================================
        # PRIORITY 7
        # PURE CONFIRMATION
        # =================================================

        if self.is_pure_yes(text):

            if (
                not state.missing_fields
                and not state.awaiting_correction
                and state.stage == "verification"
            ):

                return {
                    "intent": "confirm",
                    "action": "complete",
                    "reason": "verified_confirmation",
                    "confidence_score": 0.95
                }

        # =================================================
        # LLM FALLBACK
        # =================================================

        prompt = f"""
You are a decision engine
for a public helpline system.

Conversation State:
- stage: {state.stage}
- missing_fields: {state.missing_fields}
- awaiting_correction: {state.awaiting_correction}

User Input:
"{user_input}"

Classify into ONE:

- confirm
- correction
- restart
- unclear

Rules:
- Do NOT hallucinate
- confirm ONLY if user clearly agrees
- correction ONLY if user wants modification
- restart ONLY if user wants to restart
- otherwise unclear

Return ONLY valid JSON.

Format:
{{
    "intent": "...",
    "confidence_score": 0.0
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        parsed = self.safe_json_parse(
            response.choices[0].message.content
        )

        intent = parsed.get(
            "intent",
            "unclear"
        )

        # =================================================
        # FALLBACK: CONFIRM
        # =================================================

        if intent == "confirm":

            if (
                not state.missing_fields
                and state.stage == "verification"
            ):

                return {
                    "intent": "confirm",
                    "action": "complete",
                    "reason": "llm_confirmation",
                    "confidence_score": parsed.get(
                        "confidence_score",
                        0.8
                    )
                }

        # =================================================
        # FALLBACK: CORRECTION
        # =================================================

        if intent == "correction":

            return {
                "intent": "correction",
                "action": "ask_what_to_correct",
                "reason": "llm_correction",
                "confidence_score": parsed.get(
                    "confidence_score",
                    0.75
                )
            }

        # =================================================
        # FALLBACK: RESTART
        # =================================================

        if intent == "restart":

            return {
                "intent": "restart",
                "action": "global_reset",
                "reason": "llm_restart",
                "confidence_score": parsed.get(
                    "confidence_score",
                    0.8
                )
            }

        # =================================================
        # DEFAULT SAFE FALLBACK
        # =================================================

        return {
            "intent": "unclear",
            "action": "clarify",
            "reason": "unclear_input",
            "confidence_score": 0.5
        }

class Pipeline:

    def __init__(
        self,
        departments
    ):

        if not departments or not isinstance(departments, list):
            raise ValueError("departments must be a non-empty list")

        for dept in departments:
            if not isinstance(dept, dict):
                raise ValueError("Each department must be a dictionary")
            required_keys = {"name", "code", "required_fields"}
            if not required_keys.issubset(dept.keys()):
                raise ValueError(f"Department missing required keys: {required_keys}")

        self.input = InputAgent()

        self.u = UnderstandingAgent()
        self.c = CaseBuilderAgent(
            departments
        )

        self.i = InteractionAgent()
        self.v = VerificationAgent()
        self.d = DecisionAgent()

        self.departments = departments

    # =====================================================
    # CONTROL LAYER
    # =====================================================

    def apply_control(
        self,
        state,
        text
    ):

        t = str(text or "").lower().strip()

        human_keywords = [
            "human", "agent", "representative", "real person"
        ]

        if any(word in t for word in human_keywords):
            return {
                "action": "handover",
                "reason": "user_requested_human"
            }

        # =====================================================
        # DIRECT ATTEMPTS THRESHOLD CHECK
        # =====================================================
        # If attempts exceed max, escalate immediately
        # Don't wait for sentiment or escalation score

        if state.attempts > state.max_attempts:
            state.sentiment = "high"
            return {
                "action": "handover",
                "reason": "max_attempts_exceeded"
            }

        if self.u.detect_repetition_fatigue(state):
            state.sentiment = "high"

        is_field_drift, drift_reason = state.detect_correction_drift()
        if is_field_drift:
            state.sentiment = "high"
            if DEBUG:
                logger.debug(f"Correction drift detected: {drift_reason}")

        is_multi_pass, multi_pass_reason = state.detect_multi_pass_drift()
        if is_multi_pass:
            state.sentiment = "high"
            if DEBUG:
                logger.debug(f"Multi-pass drift detected: {multi_pass_reason}")

        exceeds_fields, exceed_reason = state.check_total_corrections_exceed_fields()
        if exceeds_fields:
            state.sentiment = "high"
            if DEBUG:
                logger.debug(f"Corrections exceed fields: {exceed_reason}")

        escalation_score = self._calculate_escalation_score(state)

        if escalation_score >= 0.75:
            return {
                "action": "handover",
                "reason": "escalation_score_critical"
            }

        if escalation_score >= 0.55 and not state.just_reset:
            return {
                "action": "reset",
                "reason": "escalation_score_high"
            }

        if (
            state.awaiting_correction
            and state.attempts >= 2
            and not state.just_reset
        ):
            return {
                "action": "reset",
                "reason": "correction_loop"
            }

        return None

    def _calculate_escalation_score(self, state):
        """Weighted scoring: (attempts*0.4) + (corrections*0.3) + (sentiment*0.3)"""
        attempt_score = min(1.0, state.attempts / state.max_attempts)
        correction_score = min(1.0, state.correction_count / float(state.max_correction_count))

        if state.sentiment == "high":
            sentiment_score = 0.7
        elif state.sentiment == "medium_high":
            sentiment_score = 0.4
        elif state.sentiment is not None:
            sentiment_score = 0.1
        else:
            sentiment_score = 0.05

        total_score = (
            attempt_score * 0.4 +
            correction_score * 0.3 +
            sentiment_score * 0.3
        )

        return total_score

    # =====================================================
    # VERIFICATION RESPONSE
    # =====================================================

    def build_verification_response(
        self,
        state
    ):

        department = (
            self.c.get_department_by_name(
                state.department
            )
        )

        if not department:

            return {
                "text": (
                    "Unable to verify the complaint."
                ),
                "action": "error"
            }

        state.set_stage("verification")

        state.last_action = "verify"

        summary = self.v.summary(
            state,
            state.field_schema
        )

        return {
            "text": summary,
            "action": "verify"
        }

    # =====================================================
    # MAIN PIPELINE
    # =====================================================

    def run(
        self,
        state,
        text_input
    ):

        # -------------------------------------------------
        # INPUT
        # -------------------------------------------------

        text = self.input.run(
            text_input=text_input
        )

        text = str(text or "").strip()

        state.last_user_message = text
        state.current_issue = text

        state.conversation_turns += 1

        # -------------------------------------------------
        # CONTROL LAYER
        # -------------------------------------------------

        control = self.apply_control(
            state,
            text
        )

        if control:

            action = control["action"]

            # ---------------------------------------------
            # HUMAN HANDOVER
            # ---------------------------------------------

            if action == "handover":

                state.escalate(
                    control["reason"]
                )

                return (
                    state,
                    self.i.human_handover(state)
                )

            # ---------------------------------------------
            # GLOBAL RESET
            # ---------------------------------------------

            if action == "reset":

                state.awaiting_new_issue = True

                # NOTE: ask_rephrase() calls soft_reset() internally
                return (
                    state,
                    self.i.ask_rephrase(state)
                )

        # =================================================
        # INITIAL CASE EXTRACTION
        # =================================================

        if (
            state.intent is None
            or state.awaiting_new_issue
        ):

            state.awaiting_new_issue = False
            state.just_reset = False

            extracted = self.u.extract(
                text,
                mode="full"
            )

            state = self.u.update_state(
                state,
                extracted
            )

            state = self.c.update_case(
                state,
                extracted,
                mode="global"
            )

            # ---------------------------------------------
            # RE-VERIFY LOW CONFIDENCE FIELDS (after soft reset)
            # ---------------------------------------------

            if state.just_reset:

                reverify_response = self.i.ask_reverify_low_confidence(state)

                if reverify_response:
                    return (state, reverify_response)

                # No low-confidence fields to reverify
                state.just_reset = False

            # ---------------------------------------------
            # ASK MISSING FIELDS
            # ---------------------------------------------

            if state.missing_fields:

                return (
                    state,
                    self.i.ask_missing(state)
                )

            # ---------------------------------------------
            # GO TO VERIFICATION
            # ---------------------------------------------

            return (
                state,
                self.build_verification_response(
                    state
                )
            )

        # =================================================
        # REVERIFY RESPONSE (after soft reset)
        # =================================================

        if state.last_action == "reverify_low_confidence":

            target_field = state.current_field

            # Check for escalation signals during reverification
            control = self.apply_control(state, text)
            if control:
                return (state, control)

            t_norm = self.d.normalize(text)

            # CASE 1: User confirms the value
            if self.d.is_pure_yes(t_norm):
                state.entity_confidence[target_field] = 0.95
                state.awaiting_reverify_confirmation = False
                state.reverify_attempts = 0
                state.last_action = None

            # CASE 2: User provides explicit correction
            elif self.d.detect_explicit_correction(t_norm):
                extracted = self.u.extract(
                    text,
                    mode="field",
                    target_field=target_field,
                    state=state
                )

                if extracted:
                    # Use reverify mode to avoid correction inflation
                    self.c.update_case(
                        state,
                        extracted,
                        mode="reverify"
                    )
                    state.entity_confidence[target_field] = 0.95
                    state.awaiting_reverify_confirmation = False
                    state.reverify_attempts = 0
                    state.last_action = None
                else:
                    # Extraction failed, ask again
                    return (
                        state,
                        self.i.ask_reverify_correction(state)
                    )

            # CASE 3: User rejects but doesn't provide correction yet
            elif self.d.is_pure_rejection(t_norm):
                # Ask user to provide the correct value
                return (
                    state,
                    self.i.ask_reverify_correction(state)
                )

            # CASE 4: Vague/uncertain response
            else:
                # Treat as rejection and ask for correction
                return (
                    state,
                    self.i.ask_reverify_correction(state)
                )

            # Check for more low-confidence fields
            reverify_response = self.i.ask_reverify_low_confidence(state)

            if reverify_response:
                return (state, reverify_response)

            # No more low-confidence fields found
            state.just_reset = False

            # Continue with missing fields
            if state.missing_fields:
                return (
                    state,
                    self.i.ask_missing(state)
                )

            # All fields done, go to verification
            return (
                state,
                self.build_verification_response(state)
            )

        # =================================================
        # REVERIFY CORRECTION RESPONSE
        # =================================================

        if state.last_action == "reverify_correction":

            target_field = state.current_field

            # Check for escalation signals
            control = self.apply_control(state, text)
            if control:
                return (state, control)

            # Check attempt limit
            if state.reverify_attempts >= state.max_reverify_attempts:
                state.sentiment = "high"
                return (state, self.i.ask_rephrase(state))

            extracted = self.u.extract(
                text,
                mode="field",
                target_field=target_field,
                state=state
            )

            # Valid correction provided
            if extracted:
                self.c.update_case(
                    state,
                    extracted,
                    mode="reverify"
                )
                state.entity_confidence[target_field] = 0.95
                state.awaiting_reverify_confirmation = False
                state.reverify_attempts = 0
                state.last_action = None

                # Check for more low-confidence fields
                reverify_response = self.i.ask_reverify_low_confidence(state)

                if reverify_response:
                    return (state, reverify_response)

                # No more low-confidence fields
                state.just_reset = False

                if state.missing_fields:
                    return (
                        state,
                        self.i.ask_missing(state)
                    )

                return (
                    state,
                    self.build_verification_response(state)
                )

            # Invalid correction, ask again
            return (
                state,
                self.i.ask_reverify_correction(state)
            )

        # =================================================
        # DECISION ENGINE
        # =================================================

        decision = self.d.decide(
            state,
            text
        )

        action = decision.get("action")

        # =================================================
        # FIELD RESPONSE
        # =================================================

        if action == "fill_field":

            target_field = (
                state.last_asked_field
            )

            extracted = self.u.extract(
                text,
                mode="field",
                target_field=target_field,
                state=state
            )

            # ---------------------------------------------
            # INVALID FIELD INPUT
            # ---------------------------------------------

            if not extracted:

                state.increment_attempts()

                state.increment_field_attempt(
                    target_field
                )

                # too many retries
                if state.exceeded_field_attempts(
                    target_field
                ):

                    return (
                        state,
                        self.i.ask_rephrase(state)
                    )

                return (
                    state,
                    self.i.ask_missing(state)
                )

            # ---------------------------------------------
            # SUCCESSFUL FIELD UPDATE
            # ---------------------------------------------

            state.reset_attempts()

            state.reset_field_attempt(
                target_field
            )

            state = self.c.update_case(
                state,
                extracted,
                mode="normal"
            )

            # Explicitly increase confidence on successful fill
            # (should already happen in update_case, but ensuring here)
            state.increase_confidence(0.02)

            # ---------------------------------------------
            # MORE FIELDS REMAIN
            # ---------------------------------------------

            if state.missing_fields:

                return (
                    state,
                    self.i.ask_missing(state)
                )

            # ---------------------------------------------
            # MOVE TO VERIFICATION
            # ---------------------------------------------

            return (
                state,
                self.build_verification_response(
                    state
                )
            )

        # =================================================
        # APPLY CORRECTION
        # CASE C + D
        # =================================================

        if action == "apply_correction":

            extracted = self.u.extract(
                text,
                mode="correction",
                state=state
            )

            # ---------------------------------------------
            # FIELD TARGET HINT
            # CASE B2
            # ---------------------------------------------

            hinted_field = extracted.get(
                "__field_hint__"
            )

            if hinted_field:

                return (
                    state,
                    self.i.ask_field_specific(
                        state,
                        hinted_field
                    )
                )

            # ---------------------------------------------
            # FAILED CORRECTION
            # ---------------------------------------------

            if not extracted:

                state.increment_attempts()

                state.reduce_confidence(
                    0.05
                )

                return (
                    state,
                    {
                        "text": (
                            "I couldn't understand "
                            "what needs correction. "
                            "Please specify clearly."
                        ),
                        "action": "correction_failed"
                    }
                )

            # ---------------------------------------------
            # APPLY CORRECTION
            # ---------------------------------------------

            state = self.c.update_case(
                state,
                extracted,
                mode="correction"
            )

            state.awaiting_correction = False

            state.reset_attempts()

            # ---------------------------------------------
            # RETURN TO VERIFICATION
            # ---------------------------------------------

            return (
                state,
                self.build_verification_response(
                    state
                )
            )

        # =================================================
        # CASE A
        # PURE REJECTION
        # =================================================

        if action == "ask_correction_or_restart":

            return (
                state,
                self.i.ask_correction_or_restart(
                    state
                )
            )

        # =================================================
        # CASE B1
        # VAGUE CORRECTION
        # =================================================

        if action == "ask_what_to_correct":

            state.awaiting_correction = True

            return (
                state,
                self.i.ask_what_to_correct(
                    state
                )
            )

        # =================================================
        # CASE B2
        # FIELD TARGET CORRECTION
        # =================================================

        if action == "ask_field_specific":

            field_name = decision.get(
                "field"
            )

            if not field_name:
                field_name = (
                    state.last_asked_field
                )

            state.awaiting_correction = True

            return (
                state,
                self.i.ask_field_specific(
                    state,
                    field_name
                )
            )

        # =================================================
        # FINAL CONFIRMATION
        # =================================================

        if action == "complete":

            if state.missing_fields:

                return (
                    state,
                    self.i.ask_missing(state)
                )

            state.is_verified = True

            state.set_stage("completed")

            state.last_action = "complete"

            return (
                state,
                {
                    "text": (
                        "Thank you. "
                        "Your complaint "
                        "has been recorded successfully."
                    ),
                    "action": "complete"
                }
            )

        # =================================================
        # GLOBAL RESET
        # =================================================

        if action == "global_reset":

            state.full_reset()

            state.awaiting_new_issue = True

            return (
                state,
                self.i.ask_rephrase(state)
            )

        # =================================================
        # DEFAULT SAFE FALLBACK
        # =================================================

        state.increment_attempts()

        return (
            state,
            {
                "text": (
                    "I'm not sure what you mean. "
                    "Could you please clarify your complaint or use 'restart' to begin a new case?"
                ),
                "action": "clarify"
            }
        )

if __name__ == "__main__":

    pipeline = Pipeline(DEPARTMENTS)
    state = CaseState()

    print("🚀 System Ready (type 'exit' to stop)\n")

    while True:

        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        # ==========================================
        # RUN PIPELINE
        # ==========================================

        state, response = pipeline.run(
            state,
            user_input
        )

        # ==========================================
        # BOT RESPONSE
        # ==========================================

        print("\nBot:", response.get("text"))

        # ==========================================
        # DEBUG PRINTS
        # ==========================================

        print("\n--- STATE ---")
        print(state)

        print("\n--- DEBUG SIGNALS ---")
        print({
            "attempts":
                state.attempts,

            "correction_count":
                state.correction_count,

            "last_corrected_field":
                state.last_corrected_field,

            "awaiting_correction":
                state.awaiting_correction,

            "just_reset":
                state.just_reset,

            "sentiment":
                state.sentiment
        })

        # ==========================================
        # SAVE DEBUG LOGS
        # ==========================================

        with open("debug_logs.txt", "a") as log:

            log.write("\n" + "="*60 + "\n")

            log.write(f"\nUser: {user_input}\n")

            log.write(
                f"\nBot: {response.get('text')}\n"
            )

            log.write("\n--- STATE ---\n")

            log.write(str(state))

            log.write("\n\n--- DEBUG SIGNALS ---\n")

            log.write(str({

                "attempts":
                    state.attempts,

                "correction_count":
                    state.correction_count,

                "last_corrected_field":
                    state.last_corrected_field,

                "awaiting_correction":
                    state.awaiting_correction,

                "just_reset":
                    state.just_reset,

                "sentiment":
                    state.sentiment

            }))

            log.write("\n")

        print("\n" + "="*60 + "\n")
