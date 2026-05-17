# Travel Planning Assistant

An AI-powered travel planning system using LangGraph, Anthropic, Gemini, and Streamlit.

## 🏗️ Architecture

The system uses a **sequential multi-agent pipeline** orchestrated by LangGraph. Each agent receives the output of the previous one via a shared state object (`TravelPlanState`).

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│              (user input + live agent progress)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               LangGraph Orchestrator (StateGraph)           │
│                   shared: TravelPlanState                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         1. Intent Agent          │
          │  Extracts destination, budget,   │
          │  duration, currency, preferences │
          └────────────────┬────────────────┘
                           │  trip_details →
          ┌────────────────▼────────────────┐
          │        2. Research Agent         │
          │  Fetches real data via           │
          │  API Ninjas + Wikipedia REST     │
          └────────────────┬────────────────┘
                           │  research_data →
          ┌────────────────▼────────────────┐
          │        3. Itinerary Agent        │
          │  Generates day-wise plan with    │
          │  activities, meals, hotels       │
          └────────────────┬────────────────┘
                           │  itinerary →
          ┌────────────────▼────────────────┐
          │         4. Budget Agent          │
          │  Estimates realistic costs in    │
          │  user's own currency             │
          └────────────────┬────────────────┘
                           │  budget_breakdown →
          ┌────────────────▼────────────────┐
          │        5. Compile Plan           │
          │  Assembles all outputs into      │
          │  final structured travel plan    │
          └────────────────┬────────────────┘
                           │  final_plan →
          ┌────────────────▼────────────────┐
          │         6. Image Agent           │
          │  Generates destination photos    │
          │  via Google Gemini (bonus)       │
          └────────────────┬────────────────┘
                           │
                           ▼
                   📋 Final Travel Plan
          (itinerary + budget + tips + images)
```

## 📋 Agents

| # | Agent | Responsibility |
|---|-------|---------------|
| 1 | **Intent Agent** | Parses user input — extracts destination, budget, duration, currency, travel style, interests |
| 2 | **Research Agent** | Fetches real destination data (weather, attractions, entry requirements) via APIs |
| 3 | **Itinerary Agent** | Builds a day-wise plan with timed activities, meal suggestions, accommodation |
| 4 | **Budget Agent** | Estimates total cost per category in the user's currency; checks against budget |
| 5 | **Orchestrator** | LangGraph `StateGraph` — manages data flow between all agents |
| 6 | **Image Agent** | Generates photorealistic destination images using Google Gemini *(bonus)* |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)
- [API Ninjas key](https://api-ninjas.com/) — for weather, city, and currency data
- [Google AI key](https://aistudio.google.com/) — for Gemini image generation

### Installation

1. Clone the repository
```bash
git clone <repo-url>
cd travel-planner
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### Running the Application

#### Streamlit UI (recommended)
```bash
streamlit run ui/streamlit_app.py
```

#### CLI Mode
```bash
python main.py
```

## 🐳 Docker

```bash
# Build
docker build -t travel-planner .

# Run
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=your_key \
  -e API_NINJAS_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  travel-planner
```

Open `http://localhost:8501`

## 📝 Sample Input / Output

**Input:**
```
Plan a 5-day budget trip to Goa under ₹25,000
```

**Output:**

```
✈️ Your Travel Plan to Goa

Trip Overview
- Duration: 5 days
- Estimated Cost: INR 23,450.00
- Daily Average: INR 4,690.00
- Budget Status: Within budget

Destination Highlights
Goa is India's smallest state, famous for its beaches, nightlife,
Portuguese heritage, and seafood cuisine.

Best Time to Visit: November to February
Weather: Tropical — warm and sunny in winter, monsoon Jun–Sep

Top Attractions
- Baga Beach
- Anjuna Flea Market
- Basilica of Bom Jesus
- Dudhsagar Falls
- Fort Aguada

Day 1: Arrival & North Goa Beaches
  09:00 AM - Check in at hostel, Calangute area
  11:00 AM - Baga Beach — swim and sunbathe (2 hrs)
  02:00 PM - Lunch at Britto's — Goan fish curry (INR 350)
  04:00 PM - Calangute Market stroll (1 hr)
  07:30 PM - Sunset at Anjuna Beach (45 mins)
  09:00 PM - Dinner at local shack — prawn masala (INR 400)
  Accommodation: zostel Goa, Calangute
  Estimated: Food INR 900 | Activities INR 200 | Transport INR 300

Budget Breakdown
- Accommodation: INR 7,500
- Food:          INR 6,500
- Activities:    INR 3,200
- Transportation:INR 3,750
- Miscellaneous: INR 2,500
─────────────────────────
Total:           INR 23,450
Status:          Within budget ✅

Money-Saving Tips
- Eat at local beach shacks instead of tourist restaurants
- Use rented scooters (INR 300/day) instead of taxis
- Book accommodation in Calangute or Anjuna (cheaper than Baga)
```

## 🎯 Key Features

- Multi-agent collaboration with real inter-agent data dependency
- Live streaming agent progress in the UI
- Real API data (API Ninjas + Wikipedia) — no hardcoded content
- Smart currency detection (₹, $, €, £, k/lac shorthand)
- Budget status calculated in code, not by the LLM
- PDF + JSON download of the final plan
- AI-generated destination images (Google Gemini)
- Session history in sidebar

## 📁 Project Structure

```
travel-planner/
├── agents/
│   ├── intent_agent.py      # Extracts trip details
│   ├── research_agent.py    # Fetches destination info
│   ├── itinerary_agent.py   # Generates day-wise plan
│   ├── budget_agent.py      # Estimates costs
│   └── image_agent.py       # Generates images (Gemini)
├── graph/
│   └── __init__.py          # LangGraph StateGraph workflow
├── models/
│   └── schemas.py           # Pydantic data models
├── tools/
│   ├── __init__.py          # API Ninjas + Wikipedia tools
│   └── pdf_generator.py     # PDF export
├── ui/
│   └── streamlit_app.py     # Streamlit frontend
├── config.py                # API keys + agent config
├── main.py                  # CLI entry point
├── Dockerfile
└── requirements.txt
```

## 🔧 Technology Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph) |
| LLM | Anthropic Claude (claude-sonnet-4) |
| Image Generation | Google Gemini |
| Research Tools | API Ninjas, Wikipedia REST API |
| UI | Streamlit |
| Data Models | Pydantic v2 |
| Language | Python 3.11+ |

## 👤 Author

P.V.S.S Sathwick
