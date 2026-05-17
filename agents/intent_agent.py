"""
Intent Agent: Extracts trip details from user input
"""
import json
import re
from langchain_anthropic import ChatAnthropic
from pydantic import ValidationError
from models.schemas import TravelPlanState, TripDetails
from config import MODEL, AGENT_CONFIG


# ---------------------------------------------------------------------------
# Only match explicit currency symbols/codes the user actually typed.
# No city or country lists — those are handled by the LLM.
# ---------------------------------------------------------------------------
_EXPLICIT_CURRENCY_RE = [
    (re.compile(r'(?:₹|(?<!\w)rs\.?(?!\w)|rupees?|(?<!\w)inr(?!\w))', re.IGNORECASE), "INR"),
    (re.compile(r'(?:\$|(?<!\w)usd(?!\w)|dollars?)',                    re.IGNORECASE), "USD"),
    (re.compile(r'(?:€|(?<!\w)eur(?!\w)|euros?)',                       re.IGNORECASE), "EUR"),
    (re.compile(r'(?:£|(?<!\w)gbp(?!\w)|pounds?)',                      re.IGNORECASE), "GBP"),
    (re.compile(r'(?:¥|(?<!\w)jpy(?!\w)|yen)',                          re.IGNORECASE), "JPY"),
    (re.compile(r'(?<!\w)aed(?!\w)|dirhams?',                           re.IGNORECASE), "AED"),
    (re.compile(r'(?<!\w)sgd(?!\w)',                                    re.IGNORECASE), "SGD"),
]


def _detect_explicit_currency(text: str) -> str | None:
    """
    Returns a currency code only when the user wrote an unambiguous symbol or code.
    Returns None for bare amounts like '25k' or '30000' with no currency hint.
    """
    for pattern, code in _EXPLICIT_CURRENCY_RE:
        if pattern.search(text):
            return code
    return None


def _normalize_numbers(text: str) -> str:
    """Convert shorthand like 25k → 25000, 1.5lac → 150000."""
    text = re.sub(
        r'\b(\d+(?:\.\d+)?)\s*[kK]\b',
        lambda m: str(int(float(m.group(1)) * 1000)),
        text,
    )
    text = re.sub(
        r'\b(\d+(?:\.\d+)?)\s*[lL](?:ac|akh)?\b',
        lambda m: str(int(float(m.group(1)) * 100000)),
        text,
    )
    return text


def _infer_currency_for_destination(destination: str, llm: ChatAnthropic) -> str:
    """
    Dedicated single-purpose LLM call: given a destination, return its currency.
    A small focused prompt is far more reliable than burying this in a large extraction prompt.
    """
    prompt = (
        f'What is the official local currency ISO 4217 code for the travel destination "{destination}"?\n'
        "Reply with ONLY the 3-letter ISO code (e.g. INR, THB, JPY, EUR). Nothing else."
    )
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        code = response.content.strip().upper()
        if re.fullmatch(r'[A-Z]{3}', code):
            return code
    except Exception:
        pass
    return "USD"  # only reached if the LLM call itself fails


def intent_agent_node(state: TravelPlanState) -> TravelPlanState:
    try:
        llm = ChatAnthropic(
            model=MODEL,
            temperature=AGENT_CONFIG["intent_agent"]["temperature"],
            max_tokens=AGENT_CONFIG["intent_agent"]["max_tokens"],
        )

        # ── Step 1: look for explicit currency symbols in raw input ───────────
        # Do this BEFORE normalization so "25k rs" still contains "rs".
        explicit_currency = _detect_explicit_currency(state.user_input)

        # ── Step 2: normalize number shorthand ────────────────────────────────
        normalized_input = _normalize_numbers(state.user_input)

        # ── Step 3: main extraction LLM call ─────────────────────────────────
        system_prompt = (
            "You are a travel planning assistant. Extract trip details from the user input "
            "and return ONLY a valid JSON object — no markdown, no extra text.\n\n"
            "Fields:\n"
            "- destination (string)\n"
            "- budget (number, symbols already stripped, shorthand already expanded; set to 0 if not mentioned)\n"
            "- currency (3-letter ISO code — your best guess; will be verified separately)\n"
            "- duration (integer days)\n"
            "- start_date (YYYY-MM-DD or null)\n"
            "- travel_style: budget / moderate / luxury\n"
            "- accommodation_type: hotel / hostel / airbnb / resort (default hotel)\n"
            "- preferences (list of strings)\n"
            "- interests (list of strings)\n\n"
            'Example: {"destination":"Goa","budget":20000,"currency":"INR","duration":5,'
            '"start_date":null,"travel_style":"budget","accommodation_type":"hostel",'
            '"preferences":["beach"],"interests":["beaches","nightlife","food"]}'
        )

        response = llm.invoke([
            {"role": "user", "content": system_prompt + f"\n\nUser input: {normalized_input}"}
        ])

        json_str = response.content.strip()
        for fence in ("```json", "```"):
            if json_str.startswith(fence):
                json_str = json_str[len(fence):]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        trip_data = json.loads(json_str)

        # ── Step 4: resolve the final currency ───────────────────────────────
        if explicit_currency:
            # User was unambiguous — trust their symbol completely.
            final_currency = explicit_currency
        else:
            # No symbol found → ask the LLM specifically about the destination.
            # A focused one-question call is far more reliable than hoping the
            # big extraction prompt gets it right for every country in the world.
            destination = trip_data.get("destination", "")
            final_currency = _infer_currency_for_destination(destination, llm)

        trip_data["currency"] = final_currency

        trip_details = TripDetails(**trip_data)
        state.trip_details = trip_details

        state.conversation_history.append({"role": "user", "content": state.user_input})
        state.conversation_history.append({
            "role": "intent_agent",
            "content": (
                f"Understood! Planning a {trip_details.duration}-day {trip_details.travel_style} trip "
                f"to {trip_details.destination} "
                f"{'with a budget of ' + trip_details.currency + ' ' + f'{trip_details.budget:,.0f}' if trip_details.budget > 0 else 'with no specified budget'}. "
                f"Interests: {', '.join(trip_details.interests) or 'general'}."
            ),
        })

    except ValidationError as e:
        state.errors.append(f"Validation error in intent agent: {str(e)}")
    except json.JSONDecodeError as e:
        state.errors.append(f"JSON parsing error in intent agent: {str(e)}")
    except Exception as e:
        state.errors.append(f"Error in intent agent: {str(e)}")

    return state