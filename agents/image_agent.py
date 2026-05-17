"""
Image Agent: Generates travel images using Gemini imagen model.
Decides 3-4 relevant image prompts based on the destination and itinerary,
generates them, saves to images/ folder, and stores paths in state.
"""
import os
import json
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from models.schemas import TravelPlanState
from config import MODEL

IMAGES_DIR = Path("images")


def _decide_image_prompts(plan, llm: ChatAnthropic) -> list[dict]:
    """
    Ask LLM to decide 3-4 relevant image prompts based on the travel plan.
    Returns list of dicts with keys: filename, prompt, caption, alt
    """
    destination = plan.destination
    attractions = plan.destination_info.attractions[:5]
    day_titles = [f"Day {d.day}: {d.title}" for d in plan.itinerary.days]
    activities = []
    for d in plan.itinerary.days:
        activities.extend(d.activities[:2])

    prompt = f"""You are a travel photography director. Based on this travel plan, decide exactly 3 images to generate.

Destination: {destination}
Top attractions: {', '.join(attractions)}
Itinerary highlights: {', '.join(day_titles)}
Key activities: {', '.join(activities[:8])}

Rules:
- Image 1: Always a stunning wide/aerial landscape shot of {destination} (hero image)
- Image 2: The most iconic attraction or landmark from the itinerary
- Image 3: A cultural/food/activity scene that captures the essence of the trip
- Image 4 (optional): If the trip has a unique highlight that would make a great photo, include it as a 4th image

For each image provide a detailed, vivid Gemini image generation prompt.
Make prompts photorealistic, specific, and visually rich. 
Always end every prompt with:
" shot on Canon 5D Mark IV, f/2.8 aperture, natural lighting,
photojournalistic style, documentary photography, 
no artificial perfection, real people real moments ".

Respond ONLY with valid JSON array:
[
    {{
        "filename": "hero_{destination.lower().replace(' ', '_')}.png",
        "prompt": "<detailed photorealistic prompt for hero landscape>",
        "caption": "<short caption>",
        "alt": "<alt text>"
    }},
    {{
        "filename": "landmark_{destination.lower().replace(' ', '_')}.png",
        "prompt": "<detailed prompt for landmark>",
        "caption": "<short caption>",
        "alt": "<alt text>"
    }},
    {{
        "filename": "culture_{destination.lower().replace(' ', '_')}.png",
        "prompt": "<detailed prompt for cultural scene>",
        "caption": "<short caption>",
        "alt": "<alt text>"
    }},
    {{
        "filename": "unique_{destination.lower().replace(' ', '_')}.png",
        "prompt": "<detailed prompt for unique highlight>",
        "caption": "<short caption>",
        "alt": "<alt text>"
    }}
]"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    json_str = response.content.strip()
    for fence in ("```json", "```"):
        if json_str.startswith(fence):
            json_str = json_str[len(fence):]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json.loads(json_str.strip())


def _generate_image_bytes(prompt: str) -> bytes:
    """
    Generate image bytes using Gemini via google-genai SDK.
    Uses gemini-2.0-flash-exp which has broad support across SDK versions.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Extract image bytes — check candidates first, then top-level parts
    candidates = getattr(resp, "candidates", None)
    parts = []
    if candidates:
        try:
            parts = candidates[0].content.parts
        except Exception:
            parts = []

    if not parts:
        parts = getattr(resp, "parts", []) or []

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in Gemini response")


def image_agent_node(state: TravelPlanState) -> TravelPlanState:
    """
    Generate 3 travel images for the destination using Gemini.
    Saves images to images/ folder and stores paths in state.
    """
    if not state.final_plan:
        state.errors.append("Final plan required for image generation")
        return state

    try:
        IMAGES_DIR.mkdir(exist_ok=True)

        llm = ChatAnthropic(model=MODEL, temperature=0.7, max_tokens=2048)

        # Step 1: Decide what images to generate
        image_specs = _decide_image_prompts(state.final_plan, llm)

        generated_images = []

        # Step 2: Generate each image
        for spec in image_specs:
            filename = spec["filename"]
            out_path = IMAGES_DIR / filename

            try:
                if not out_path.exists():
                    img_bytes = _generate_image_bytes(spec["prompt"])
                    out_path.write_bytes(img_bytes)

                generated_images.append({
                    "path": str(out_path),
                    "filename": filename,
                    "caption": spec["caption"],
                    "alt": spec["alt"],
                    "prompt": spec["prompt"],
                })

            except Exception as e:
                # Graceful fallback — skip this image, don't fail the whole plan
                state.errors.append(f"Image generation failed for {filename}: {str(e)}")
                continue

        # Step 3: Store in state
        state.generated_images = generated_images

        state.conversation_history.append({
            "role": "image_agent",
            "content": (
                f"Generated {len(generated_images)} images for {state.final_plan.destination}: "
                f"{', '.join(img['caption'] for img in generated_images)}."
            ),
        })

    except Exception as e:
        state.errors.append(f"Error in image agent: {str(e)}")

    return state