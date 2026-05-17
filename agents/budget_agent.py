"""
Budget Agent: Estimates realistic trip costs in the user's own currency.
No USD round-trip conversion — the LLM reasons in the destination's native
price context and outputs amounts directly in user_currency.
"""
import json
from langchain_anthropic import ChatAnthropic
from pydantic import ValidationError
from models.schemas import TravelPlanState, BudgetBreakdown
from config import MODEL, AGENT_CONFIG


def budget_agent_node(state: TravelPlanState) -> TravelPlanState:
    if not state.trip_details or not state.itinerary:
        state.errors.append("Trip details and itinerary required for budget calculation")
        return state

    try:
        llm = ChatAnthropic(
            model=MODEL,
            temperature=AGENT_CONFIG["budget_agent"]["temperature"],
            max_tokens=AGENT_CONFIG["budget_agent"]["max_tokens"],
        )

        trip = state.trip_details
        user_currency = trip.currency.upper()
        destination = trip.destination
        duration = trip.duration
        travel_style = trip.travel_style
        user_budget = trip.budget  # already in user's currency — use as-is

        research_context = ""
        if state.research_data:
            research_context = (
                f"\nDestination context:\n"
                f"- Overview: {state.research_data.overview[:300]}\n"
                f"- Transportation: {state.research_data.transportation}\n"
                f"- Tips: {', '.join(state.research_data.travel_tips[:3])}\n"
            )

        activities_context = "\n".join(
            f"Day {d.day}: {', '.join(d.activities[:3])}"
            for d in state.itinerary.days
        )

        system_prompt = f"""You are a travel budget expert. Estimate a REALISTIC cost breakdown for this trip.

Trip details:
- Destination: {destination}
- Duration: {duration} days
- Travel style: {travel_style}
- User's total budget: {user_currency} {user_budget:,.0f}
- Output currency: {user_currency}

IMPORTANT: All amounts in your response must be in {user_currency}.
Use your knowledge of actual local prices in {destination} — think in the local economy,
not in USD. For example, if the destination is in India, think in rupees; if Thailand, in baht, etc.

Planned activities to cost:
{activities_context}
{research_context}
Estimate totals for the FULL {duration}-day trip (not per day) across these categories:
- accommodation: total for {duration} nights
- food: total for {duration} days (mix of local eateries and occasional restaurants)
- activities: total entry fees, tours, experiences from the itinerary
- transportation: total local transport (auto, metro, cab, intercity buses/trains)
- miscellaneous: shopping, tips, buffer for unexpected costs

Respond ONLY with valid JSON, no markdown:
{{
    "destination": "{destination}",
    "duration": {duration},
    "currency": "{user_currency}",
    "categories": {{
        "accommodation": <number>,
        "food": <number>,
        "activities": <number>,
        "transportation": <number>,
        "miscellaneous": <number>
    }},
    "total_estimated": <sum of all categories>,
    "per_day_average": <total_estimated / {duration}>,
    "budget_status": "within budget or over budget placeholder — will be recalculated",
    "recommendations": [<3-5 specific money-saving tips for {destination}>]
}}"""

        response = llm.invoke([{"role": "user", "content": system_prompt}])
        json_str = response.content.strip()

        for fence in ("```json", "```"):
            if json_str.startswith(fence):
                json_str = json_str[len(fence):]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        budget_dict = json.loads(json_str)

        # Enforce currency — LLM must not have overridden it
        budget_dict["currency"] = user_currency

        # Recalculate totals from categories to be safe (in case LLM math drifts)
        categories = budget_dict.get("categories", {})
        total = round(sum(categories.values()), 2)
        budget_dict["total_estimated"] = total
        budget_dict["per_day_average"] = round(total / duration, 2)

        # Authoritative budget status check — done in code, not by the LLM
        if user_budget == 0:
            budget_dict["budget_status"] = "Estimated"
        elif total <= user_budget:
            budget_dict["budget_status"] = "Within budget"
        else:
            overage = total - user_budget
            budget_dict["budget_status"] = f"Over budget by {user_currency} {overage:,.0f}"

        budget_breakdown = BudgetBreakdown(**budget_dict)
        state.budget_breakdown = budget_breakdown
        state.conversation_history.append({
            "role": "budget_agent",
            "content": (
                f"Estimated total cost: {user_currency} {budget_breakdown.total_estimated:,.0f} "
                f"({user_currency} {budget_breakdown.per_day_average:,.0f}/day). "
                f"Status: {budget_breakdown.budget_status}."
            ),
        })

    except ValidationError as e:
        state.errors.append(f"Validation error in budget agent: {str(e)}")
    except json.JSONDecodeError as e:
        state.errors.append(f"JSON parsing error in budget agent: {str(e)}")
    except Exception as e:
        state.errors.append(f"Error in budget agent: {str(e)}")

    return state