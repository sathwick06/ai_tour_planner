"""
LangGraph workflow for coordinating travel planning agents
"""
from langgraph.graph import StateGraph
from models.schemas import TravelPlanState, FinalTravelPlan
from agents import (
    intent_agent_node,
    research_agent_node,
    itinerary_agent_node,
    budget_agent_node,
    image_agent_node,
)
from datetime import datetime


def create_travel_planner_graph():
    """
    Create and configure the LangGraph workflow
    
    Returns:
        Compiled workflow graph
    """
    
    # Create the StateGraph
    graph = StateGraph(TravelPlanState)
    
    # Add nodes
    graph.add_node("intent_agent", intent_agent_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("itinerary_agent", itinerary_agent_node)
    graph.add_node("budget_agent", budget_agent_node)
    graph.add_node("compile_plan", compile_final_plan)
    graph.add_node("image_agent", image_agent_node)
    
    # Define edges (workflow)
    graph.set_entry_point("intent_agent")
    graph.add_edge("intent_agent", "research_agent")
    graph.add_edge("research_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "budget_agent")
    graph.add_edge("budget_agent", "compile_plan")
    graph.add_edge("compile_plan", "image_agent")
    graph.set_finish_point("image_agent")
    
    # Compile the graph
    return graph.compile()


def compile_final_plan(state: TravelPlanState) -> TravelPlanState:
    """
    Compile all agent outputs into final travel plan
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with final plan
    """
    
    try:
        if (
            state.trip_details
            and state.research_data
            and state.itinerary
            and state.budget_breakdown
        ):
            final_plan = FinalTravelPlan(
                destination=state.trip_details.destination,
                duration=state.trip_details.duration,
                trip_details=state.trip_details,
                destination_info=state.research_data,
                itinerary=state.itinerary,
                budget=state.budget_breakdown,
                generated_at=datetime.now().isoformat(),
            )
            state.final_plan = final_plan
        else:
            state.errors.append("Missing required data to compile final plan")

    except Exception as e:
        state.errors.append(f"Error compiling final plan: {str(e)}")

    if state.final_plan:
        state.conversation_history.append({
            "role": "compile_plan",
            "content": (
                f"Plan compiled for {state.final_plan.destination} — "
                f"{state.final_plan.duration} days, "
                f"{state.final_plan.budget.currency} {state.final_plan.budget.total_estimated:,.0f} total. "
                f"Ready for image generation."
            ),
        })

    return state


# Create the workflow instance
travel_planner_graph = create_travel_planner_graph()
