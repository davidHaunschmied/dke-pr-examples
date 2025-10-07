import os
import json
import urllib.request
import urllib.parse
import google.generativeai as genai

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# Weather API
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# City coordinates
CITIES = {
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "rome": (41.9028, 12.4964),
    "moscow": (55.7558, 37.6173)
}


def retrieve_weather(city, latitude, longitude):
    """Retrieve weather data"""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "hourly": "temperature_2m,precipitation_probability",
        "forecast_days": 1
    }

    url = WEATHER_API_URL + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    context = f"""Weather in {city}:
- Temperature: {current.get('temperature_2m')}°C
- Humidity: {current.get('relative_humidity_2m')}%
- Wind speed: {current.get('wind_speed_10m')} km/h
- Next hours temperatures: {hourly.get('temperature_2m', [])[:6]}
- Precipitation probability: {hourly.get('precipitation_probability', [])[:6]}%
"""
    return context


def rag_query(question, city, latitude, longitude):
    """RAG: Retrieve weather and generate answer"""

    # Step 1: Retrieve
    print("\nRetrieving weather data...")
    context = retrieve_weather(city, latitude, longitude)
    print(context)

    # Step 2: Generate
    print("\nGenerating answer...")
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")

    prompt = f"""Based on this weather data, answer the question.

{context}

Question: {question}"""

    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    print("=== Weather RAG System ===\n")

    # Get city from user
    print("Available cities:", ", ".join(CITIES.keys()))
    city_input = input("\nEnter city name: ").strip().lower()

    if city_input not in CITIES:
        print(f"City '{city_input}' not found. Using London as default.")
        city_input = "london"

    lat, lon = CITIES[city_input]

    # Get question from user
    question = input("\nAsk a question about the weather: ").strip()

    if not question:
        question = "What's the weather like?"

    # Run RAG
    answer = rag_query(question, city_input, lat, lon)

    print("\n=== Answer ===")
    print(answer)