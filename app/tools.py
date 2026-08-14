"""The 4 dynamic-data tools from the brief, mocked with fixed in-memory data.

Swapping these bodies for real API calls changes nothing else in the system. The
descriptions the LLM actually reads live in router.py's TOOLS_SCHEMA, not here.
"""

_FLIGHTS_DB = {
    "AH1235": {
        "status": "cancelled",
        "scheduled_departure": "08:30",
        "estimated_departure": None,
        "arrival_time": None,
        "terminal": "2B",
        "gate": "14",
    },
    "AH1009": {
        "status": "on time",
        "scheduled_departure": "14:00",
        "estimated_departure": "14:00",
        "arrival_time": "16:15",
        "terminal": "1",
        "gate": "22",
    },
}

_AIRPORTS_DB = {
    "CDG": {
        "name": "Charles de Gaulle",
        "city": "Paris",
        "terminals": ["1", "2A-2G", "3"],
        "timezone": "Europe/Paris",
    },
    "ALG": {
        "name": "Houari Boumediene",
        "city": "Algiers",
        "terminals": ["1", "2"],
        "timezone": "Africa/Algiers",
    },
    "CMN": {
        "name": "Mohammed V International Airport",
        "city": "Casablanca",
        "terminals": ["1", "2"],
        "timezone": "Africa/Casablanca",
    },
}

_BOOKINGS_DB = {
    "ABC123": {
        "flight_number": "AH1235",
        "date": "2026-08-12",
        "passengers": 1,
        "class": "Economy",
        "origin": "Paris CDG",
        "destination": "Algiers",
        "baggage": "1 checked bag (23kg)",
    },
}


def search_flights(origin: str, destination: str, departure_date: str) -> dict:
    return {
        "flights": [
            {
                "flight_number": "AH1235",
                "origin": origin,
                "destination": destination,
                "departure_time": "08:30",
                "arrival_time": "10:45",
                "price": "250 EUR",
            }
        ]
    }


def get_flight_status(flight_number: str, date: str) -> dict:
    return _FLIGHTS_DB.get(flight_number, {"error": f"No data found for flight {flight_number}"})


def get_airport_info(airport_code: str) -> dict:
    return _AIRPORTS_DB.get(airport_code, {"error": f"No data found for airport {airport_code}"})


def get_booking(booking_reference: str) -> dict:
    return _BOOKINGS_DB.get(booking_reference, {"error": f"No booking found for reference {booking_reference}"})


TOOL_FUNCTIONS = {
    "search_flights": search_flights,
    "get_flight_status": get_flight_status,
    "get_airport_info": get_airport_info,
    "get_booking": get_booking,
}
