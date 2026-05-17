"""
Research tools using API Ninjas + Wikipedia REST API
"""
import requests
from typing import Dict, Any, List
from config import API_NINJAS_KEY

_NINJAS_HEADERS = {"X-Api-Key": API_NINJAS_KEY}
_NINJAS_BASE = "https://api.api-ninjas.com/v1"
_WIKI_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _get_city_info(city: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{_NINJAS_BASE}/city",
            params={"name": city, "limit": 1},
            headers=_NINJAS_HEADERS,
            timeout=10,
        )
        results = resp.json()
        return results[0] if results else {}
    except Exception:
        return {}


def _get_weather(city: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{_NINJAS_BASE}/weather",
            params={"city": city},
            headers=_NINJAS_HEADERS,
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def _get_country_info(country: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{_NINJAS_BASE}/country",
            params={"name": country},
            headers=_NINJAS_HEADERS,
            timeout=10,
        )
        results = resp.json()
        return results[0] if results else {}
    except Exception:
        return {}


def _get_wikipedia_summary(query: str) -> str:
    try:
        slug = query.strip().replace(" ", "_")
        resp = requests.get(f"{_WIKI_BASE}/{slug}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("extract", "")
        # Fallback: search endpoint
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=10,
        )
        results = search_resp.json().get("query", {}).get("search", [])
        if results:
            title = results[0]["title"].replace(" ", "_")
            r2 = requests.get(f"{_WIKI_BASE}/{title}", timeout=10)
            if r2.status_code == 200:
                return r2.json().get("extract", "")
        return ""
    except Exception:
        return ""


def get_destination_info(destination: str) -> str:
    """
    Fetch real destination data from API Ninjas and Wikipedia.
    Returns a formatted string consumed by the research agent.
    """
    city_info = _get_city_info(destination)
    weather = _get_weather(destination)
    country_name = city_info.get("country", "")
    country_info = _get_country_info(country_name) if country_name else {}
    wiki_summary = _get_wikipedia_summary(destination)

    sections: List[str] = [f"Destination: {destination}"]

    if wiki_summary:
        sections.append(f"\nOverview:\n{wiki_summary}")

    if city_info:
        pop = city_info.get("population", "N/A")
        pop_str = f"{pop:,}" if isinstance(pop, int) else str(pop)
        sections.append(
            f"\nCity Facts:"
            f"\n- Country: {city_info.get('country', 'N/A')}"
            f"\n- Population: {pop_str}"
            f"\n- Is Capital: {city_info.get('is_capital', False)}"
        )

    if weather:
        temp_c = round(weather.get("temp", 0) - 273.15, 1) if weather.get("temp", 0) > 100 else weather.get("temp", "N/A")
        min_c = round(weather.get("min_temp", 0) - 273.15, 1) if weather.get("min_temp", 0) > 100 else weather.get("min_temp", "N/A")
        max_c = round(weather.get("max_temp", 0) - 273.15, 1) if weather.get("max_temp", 0) > 100 else weather.get("max_temp", "N/A")
        sections.append(
            f"\nCurrent Weather:"
            f"\n- Temperature: {temp_c}°C (min: {min_c}°C, max: {max_c}°C)"
            f"\n- Humidity: {weather.get('humidity', 'N/A')}%"
            f"\n- Wind Speed: {weather.get('wind_speed', 'N/A')} m/s"
        )

    if country_info:
        gdp = country_info.get("gdp_per_capita_usd", "N/A")
        sections.append(
            f"\nCountry Information:"
            f"\n- Currency: {country_info.get('currency', {}).get('name', 'N/A')} ({country_info.get('currency', {}).get('code', 'N/A')})"
            f"\n- Languages: {', '.join(country_info.get('languages', ['N/A']))}"
            f"\n- GDP per capita: ${gdp}"
            f"\n- Region: {country_info.get('region', 'N/A')}"
        )

    # Fetch attractions via Wikipedia search
    attractions_summary = _get_wikipedia_summary(f"tourist attractions {destination}")
    if attractions_summary:
        sections.append(f"\nAttractions & Tourism:\n{attractions_summary[:600]}")

    return "\n".join(sections)


def search_attractions(destination: str, category: str = "all") -> List[str]:
    """
    Search attractions using Wikipedia. Returns a list of attraction names.
    """
    query = f"{category} attractions {destination}" if category != "all" else f"tourist attractions {destination}"
    summary = _get_wikipedia_summary(query)
    if not summary:
        return []
    # Return the summary as a single rich text item for the agent to parse
    return [summary[:800]]


def estimate_costs(destination: str, travel_style: str) -> Dict[str, float]:
    """
    Estimate daily costs using country GDP data as a scaling factor.
    Falls back to sensible defaults if API data is unavailable.
    """
    city_info = _get_city_info(destination)
    country_name = city_info.get("country", "")
    country_info = _get_country_info(country_name) if country_name else {}

    gdp = country_info.get("gdp_per_capita_usd", 15000)
    try:
        gdp = float(gdp)
    except (TypeError, ValueError):
        gdp = 15000

    # Scale factor relative to a $15k GDP baseline
    scale = min(max(gdp / 15000, 0.3), 4.0)

    base = {
        "budget":   {"accommodation": 25, "food": 15, "activities": 10, "transport": 8},
        "moderate": {"accommodation": 70, "food": 40, "activities": 30, "transport": 18},
        "luxury":   {"accommodation": 200, "food": 120, "activities": 80, "transport": 45},
    }.get(travel_style, {"accommodation": 70, "food": 40, "activities": 30, "transport": 18})

    return {k: round(v * scale, 2) for k, v in base.items()}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """
    Convert an amount between currencies using API Ninjas.
    Returns the converted amount, or the original amount if conversion fails.
    """
    if from_currency.upper() == to_currency.upper():
        return round(amount, 2)
    try:
        resp = requests.get(
            f"{_NINJAS_BASE}/convertcurrency",
            params={"have": from_currency.upper(), "want": to_currency.upper(), "amount": amount},
            headers=_NINJAS_HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            return round(resp.json().get("new_amount", amount), 2)
        return round(amount, 2)
    except Exception:
        return round(amount, 2)


__all__ = ["get_destination_info", "search_attractions", "estimate_costs", "convert_currency"]
