"""
Itinerary Agent: Generates day-wise travel plan
"""
import json
from langchain_anthropic import ChatAnthropic
from pydantic import ValidationError
from models.schemas import TravelPlanState, ItineraryPlan, DayPlan
from config import MODEL, AGENT_CONFIG


def itinerary_agent_node(state: TravelPlanState) -> TravelPlanState:
    """
    Generate detailed day-wise itinerary
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with itinerary
    """
    
    if not state.trip_details or not state.research_data:
        state.errors.append("Trip details and research data required for itinerary generation")
        return state
    
    try:
        # Initialize Claude model
        max_tokens = max(2000, state.trip_details.duration * 800)
        llm = ChatAnthropic(
            model=MODEL,
            temperature=AGENT_CONFIG["itinerary_agent"]["temperature"],
            max_tokens=max_tokens,
        )
        
        # Prepare context
        trip_details = state.trip_details.model_dump()
        research_data = state.research_data.model_dump()
        currency = state.trip_details.currency
        travel_style = state.trip_details.travel_style

        system_prompt = f"""You are an expert travel itinerary planner. Create a detailed day-wise travel plan
based on the provided trip details and destination research.

IMPORTANT: All cost estimates in the "estimates" field must be realistic amounts in {currency}.
Use actual local market prices for {state.trip_details.destination}.
Travel style is {travel_style} — scale costs accordingly.

For each day include:
- day: Day number
- title: Catchy theme title
- activities: 6-8 specific activities with exact timing in format "HH:MM AM/PM - Activity name and description (duration)".
  Cover morning, afternoon, and evening slots.
- meals: Restaurant name + cuisine type + approx cost in {currency} (breakfast, lunch, dinner)
- accommodation: Hotel/hostel name with area/neighborhood
- estimates: Realistic per-day costs in {currency} (food, activities, transport)

Respond ONLY with valid JSON:
{{
    "destination": "destination_name",
    "duration": 5,
    "days": [
        {{
            "day": 1,
            "title": "Arrival & City Orientation",
            "activities": [
                "09:00 AM - Visit XYZ temple, explore the main shrine and gardens (1.5 hrs)",
                "11:00 AM - Walk through ABC market, try local street food (1 hr)",
                "02:00 PM - Tour DEF museum, focus on historical exhibits (2 hrs)",
                "05:00 PM - Sunset at GHI viewpoint, great photo opportunity (45 mins)",
                "07:30 PM - Evening stroll at JKL promenade (1 hr)"
            ],
            "meals": {{
                "breakfast": "Hotel restaurant - South Indian buffet - INR 200",
                "lunch": "Cafe XYZ - Local thali cuisine - INR 300",
                "dinner": "Restaurant ABC - Seafood specialties - INR 600"
            }},
            "accommodation": "Hotel Name, Neighborhood Area",
            "estimates": {{"food": 800, "activities": 200, "transport": 300}}
        }}
    ],
    "special_notes": ["Tip 1", "Tip 2"]
}}"""
        
        messages = [
            {
                "role": "user",
                "content": system_prompt + f"\n\nTrip Details:\n{json.dumps(trip_details, indent=2)}\n\nDestination Info:\n{json.dumps(research_data, indent=2)}"
            }
        ]
        
        # Call Claude API
        response = llm.invoke(messages)
        response_text = response.content
        
        # Parse JSON response
        json_str = response_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        itinerary_dict = json.loads(json_str)
        
        # Convert to ItinerayPlan
        days = [DayPlan(**day) for day in itinerary_dict.get("days", [])]
        
        itinerary = ItineraryPlan(
            destination=itinerary_dict.get("destination", state.trip_details.destination),
            duration=itinerary_dict.get("duration", state.trip_details.duration),
            days=days,
            special_notes=itinerary_dict.get("special_notes", [])
        )
        
        state.itinerary = itinerary
        day_titles = ", ".join(f"Day {d.day}: {d.title}" for d in days[:3])
        state.conversation_history.append({
            "role": "itinerary_agent",
            "content": (
                f"Created a {itinerary.duration}-day itinerary for {itinerary.destination}. "
                f"{day_titles}{'...' if len(days) > 3 else ''}. "
                f"Special notes: {'; '.join(itinerary.special_notes[:2]) or 'None'}."
            ),
        })

    except ValidationError as e:
        state.errors.append(f"Validation error in itinerary agent: {str(e)}")
    except json.JSONDecodeError as e:
        state.errors.append(f"JSON parsing error in itinerary agent: {str(e)}")
    except Exception as e:
        state.errors.append(f"Error in itinerary agent: {str(e)}")
    
    return state