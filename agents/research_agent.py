"""
Research Agent: Fetches destination information
"""
import json
from langchain_anthropic import ChatAnthropic
from pydantic import ValidationError
from models.schemas import TravelPlanState, ResearchData
from config import MODEL, AGENT_CONFIG
from tools import get_destination_info


def research_agent_node(state: TravelPlanState) -> TravelPlanState:
    """
    Research destination and gather information
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with research data
    """
    
    if not state.trip_details:
        state.errors.append("Trip details not available for research agent")
        return state
    
    try:
        # Get destination information
        destination = state.trip_details.destination
        
        # Use research tools to fetch data
        research_content = get_destination_info(destination)
        
        # Initialize Claude model
        llm = ChatAnthropic(
            model=MODEL,
            temperature=AGENT_CONFIG["research_agent"]["temperature"],
            max_tokens=AGENT_CONFIG["research_agent"]["max_tokens"],
        )
        
        # System prompt for research compilation
        system_prompt = """You are a travel researcher. Based on the provided information about a destination, 
create a comprehensive research summary.

Respond ONLY with a valid JSON object with these fields:
- destination: Name of destination
- overview: Brief overview of the destination
- best_time_to_visit: Best season/months to visit
- weather: General weather patterns
- attractions: List of top attractions (create if not found)
- cuisine: Local cuisine highlights
- transportation: Local transportation options
- entry_requirements: Visa/entry requirements
- travel_tips: List of practical travel tips

If exact information is not available, use general knowledge and context to provide helpful information.

Example response:
{
    "destination": "Paris",
    "overview": "The City of Light, famous for art, culture, and romance",
    "best_time_to_visit": "April-May and September-October",
    "weather": "Mild springs, warm summers, cool autumns",
    "attractions": ["Eiffel Tower", "Louvre Museum", "Notre-Dame"],
    "cuisine": "French cuisine, pastries, wines",
    "transportation": "Metro, buses, trains",
    "entry_requirements": "EU citizens: ID card, Others: Passport valid 6+ months",
    "travel_tips": ["Learn basic French phrases", "Book museums in advance", "Use public transport"]
}"""
        
        messages = [
            {
                "role": "user",
                "content": system_prompt + f"\n\nDestination Information:\n{research_content}"
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
        
        research_data_dict = json.loads(json_str)
        research_data = ResearchData(**research_data_dict)
        
        state.research_data = research_data
        state.conversation_history.append({
            "role": "research_agent",
            "content": (
                f"Researched {research_data.destination}. "
                f"Best time to visit: {research_data.best_time_to_visit}. "
                f"Top attractions: {', '.join(research_data.attractions[:3])}. "
                f"Weather: {research_data.weather}."
            ),
        })

    except ValidationError as e:
        state.errors.append(f"Validation error in research agent: {str(e)}")
    except json.JSONDecodeError as e:
        state.errors.append(f"JSON parsing error in research agent: {str(e)}")
    except Exception as e:
        state.errors.append(f"Error in research agent: {str(e)}")
    
    return state
