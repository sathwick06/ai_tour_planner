"""
Streamlit UI for Travel Planning Assistant
- Streaming: agent status cards appear one by one as each agent completes
- History: sidebar shows all plans generated in this session
- Images: Gemini-generated destination images shown in the plan
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import json
from datetime import datetime
from models.schemas import TravelPlanState
from graph import travel_planner_graph
from config import STREAMLIT_CONFIG
from tools.pdf_generator import generate_travel_plan_pdf


# ── Agent metadata ────────────────────────────────────────────────────────────
AGENTS = [
    ("intent_agent",    "🎯", "Intent Agent"),
    ("research_agent",  "🔍", "Research Agent"),
    ("itinerary_agent", "🗓️", "Itinerary Agent"),
    ("budget_agent",    "💰", "Budget Agent"),
    ("compile_plan",    "📋", "Compiling Plan"),
    ("image_agent",     "📸", "Image Agent"),
]


# ── Format final plan as markdown ─────────────────────────────────────────────
def format_travel_plan(state: TravelPlanState) -> str:
    if not state.final_plan:
        return "No travel plan generated."

    plan = state.final_plan
    currency = (
        getattr(plan.budget, "currency", None)
        or getattr(plan.trip_details, "currency", None)
        or "USD"
    )

    output = f"""
# ✈️ Your Travel Plan to {plan.destination}

## Trip Overview
- **Duration:** {plan.duration} days
- **Estimated Cost:** {currency} {plan.budget.total_estimated:,.2f}
- **Daily Average:** {currency} {plan.budget.per_day_average:,.2f}
- **Budget Status:** {plan.budget.budget_status}

## Destination Highlights
**{plan.destination_info.overview}**

### Best Time to Visit
{plan.destination_info.best_time_to_visit}

### Weather
{plan.destination_info.weather}

### Top Attractions
"""
    for attraction in plan.destination_info.attractions:
        output += f"- {attraction}\n"

    output += f"""
### Local Cuisine
{plan.destination_info.cuisine}

### Transportation
{plan.destination_info.transportation}

---

## Day-by-Day Itinerary

"""
    for day in plan.itinerary.days:
        output += f"### Day {day.day}: {day.title}\n\n**Activities:**\n"
        for activity in day.activities:
            output += f"- {activity}\n"
        output += "\n**Meals:**\n"
        for meal_type, restaurant in day.meals.items():
            output += f"- {meal_type.capitalize()}: {restaurant}\n"
        output += f"\n**Accommodation:** {day.accommodation}\n\n**Estimated Costs:**\n"
        for cost_type, amount in day.estimates.items():
            output += f"- {cost_type.capitalize()}: {currency} {amount:,.2f}\n"
        output += "\n---\n"

    output += "\n## Budget Breakdown\n\n"
    for category, amount in plan.budget.categories.items():
        output += f"- **{category.capitalize()}:** {currency} {amount:,.2f}\n"

    output += f"""
### Total Estimated Cost: {currency} {plan.budget.total_estimated:,.2f}

**Status:** {plan.budget.budget_status}

### Money-Saving Tips
"""
    for tip in plan.budget.recommendations:
        output += f"- {tip}\n"

    output += "\n## Travel Tips\n"
    for tip in plan.destination_info.travel_tips:
        output += f"- {tip}\n"

    if plan.itinerary.special_notes:
        output += "\n## Special Notes\n"
        for note in plan.itinerary.special_notes:
            output += f"- {note}\n"

    return output


def display_images(images: list) -> None:
    """Display generated images in a nice grid."""
    if not images:
        return

    st.markdown("## 📸 Destination Gallery")
    cols = st.columns(len(images))
    for i, img in enumerate(images):
        path = Path(img["path"])
        if path.exists():
            with cols[i]:
                st.image(str(path), caption=img["caption"], use_container_width=True)


# ── Run pipeline with live streaming status cards ─────────────────────────────
def run_with_streaming(user_input: str) -> TravelPlanState:
    state = TravelPlanState(user_input=user_input)

    st.markdown("#### 🤖 Agent Progress")

    # Create placeholder cards for all steps upfront
    placeholders = {}
    for key, icon, label in AGENTS:
        placeholders[key] = st.empty()
        placeholders[key].info(f"{icon} **{label}:** ⏳ Waiting...")

    # Stream through graph — each node emits its result as it completes
    for step in travel_planner_graph.stream(state.model_dump()):
        node_name = list(step.keys())[0]
        node_state = TravelPlanState(**list(step.values())[0])

        if node_name not in placeholders:
            continue

        icon  = next(i for k, i, l in AGENTS if k == node_name)
        label = next(l for k, i, l in AGENTS if k == node_name)

        message = next(
            (m["content"] for m in reversed(node_state.conversation_history) if m["role"] == node_name),
            "Completed."
        )

        if node_state.errors:
            placeholders[node_name].warning(f"{icon} **{label}:** ✅ {message}")
        else:
            placeholders[node_name].success(f"{icon} **{label}:** ✅ {message}")

        state = node_state

    return state


# ── Sidebar: session history ──────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.header("✈️ Travel Planner")
        st.markdown("*AI-powered trip planning*")
        st.divider()

        if st.session_state.history:
            st.subheader("📋 Previous Plans")
            for i, entry in enumerate(reversed(st.session_state.history)):
                idx = len(st.session_state.history) - 1 - i
                destination = entry["destination"]
                currency = entry["currency"]
                total = entry["total"]
                duration = entry["duration"]
                timestamp = entry["timestamp"]

                if st.button(
                    f"🗺️ {destination} — {currency} {total:,.0f} ({duration}d)\n_{timestamp}_",
                    key=f"history_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.viewing_history = idx
                    st.rerun()

            st.divider()

        st.info(
            """
**Features:**
- Destination research
- Personalized itineraries
- Budget estimation
- AI-generated images
- Travel tips
            """
        )


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
        layout=STREAMLIT_CONFIG["layout"],
    )

    # ── Session state init ────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    if "viewing_history" not in st.session_state:
        st.session_state.viewing_history = None

    render_sidebar()

    # ── History view ──────────────────────────────────────────────────────────
    if st.session_state.viewing_history is not None:
        idx = st.session_state.viewing_history
        entry = st.session_state.history[idx]

        st.title(f"✈️ Saved Plan: {entry['destination']}")
        st.caption(f"Generated at {entry['timestamp']}")

        if st.button("⬅️ Back to Planner"):
            st.session_state.viewing_history = None
            st.rerun()

        # Show saved images
        display_images(entry["state"].generated_images)

        st.markdown(format_travel_plan(entry["state"]))

        col1, col2 = st.columns(2)
        with col1:
            plan_json = json.dumps(entry["state"].final_plan.model_dump(), indent=2, default=str)
            st.download_button(
                "📥 Download (JSON)", plan_json,
                file_name=f"travel_plan_{entry['destination']}.json",
                mime="application/json", use_container_width=True,
            )
        with col2:
            try:
                pdf_data = generate_travel_plan_pdf(
                    entry["state"].final_plan,
                    entry["state"].generated_images,
                )
                st.download_button(
                    "📄 Download (PDF)", pdf_data,
                    file_name=f"travel_plan_{entry['destination']}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")
        return

    # ── Main planner UI ───────────────────────────────────────────────────────
    st.title("✈️ Travel Planning Assistant")
    st.markdown("*Plan your perfect trip with AI-powered personalized recommendations*")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🏖️ Tell us about your trip")
        user_input = st.text_area(
            "Describe your travel plans",
            placeholder="E.g., 'Plan a 5-day budget trip to Goa under ₹25k with beach activities'",
            height=120,
            key="user_input",
        )
    with col2:
        st.subheader("💡 Tips")
        st.markdown(
            """
- Be specific about:
  - Destination
  - Budget (₹/$/€)
  - Duration
  - Interests
- Mention travel style:
  - Budget / Moderate / Luxury
            """
        )

    if st.button("🚀 Generate Travel Plan", use_container_width=True):
        if not user_input.strip():
            st.error("❌ Please describe your travel plans")
        else:
            st.divider()
            try:
                result = run_with_streaming(user_input)

                if result.errors and not result.final_plan:
                    st.error("❌ Could not generate plan. See agent status above.")
                else:
                    st.session_state.current_state = result

                    plan = result.final_plan
                    currency = getattr(plan.budget, "currency", "USD")
                    st.session_state.history.append({
                        "destination": plan.destination,
                        "duration": plan.duration,
                        "currency": currency,
                        "total": plan.budget.total_estimated,
                        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
                        "state": result,
                    })

                    st.success("✅ Travel plan generated successfully!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ── Display current plan ──────────────────────────────────────────────────
    if st.session_state.current_state and st.session_state.current_state.final_plan:
        st.divider()
        state = st.session_state.current_state

        if state.errors:
            with st.expander("⚠️ Warnings"):
                for error in state.errors:
                    st.warning(error)

        # Show generated images first
        display_images(state.generated_images)

        st.markdown(format_travel_plan(state))

        col1, col2 = st.columns(2)
        with col1:
            plan_json = json.dumps(state.final_plan.model_dump(), indent=2, default=str)
            st.download_button(
                "📥 Download (JSON)", plan_json,
                file_name=f"travel_plan_{state.final_plan.destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json", use_container_width=True,
            )
        with col2:
            try:
                pdf_data = generate_travel_plan_pdf(
                    state.final_plan,
                    state.generated_images,
                )
                st.download_button(
                    "📄 Download (PDF)", pdf_data,
                    file_name=f"travel_plan_{state.final_plan.destination}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center'><p style='color:gray;'>"
        "Made with ❤️ using LangGraph, Anthropic, Gemini, and Streamlit"
        "</p></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()