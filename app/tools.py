"""The 4 dynamic-data tools from the challenge brief.

These are mocked: no real airline API is available for this challenge, so each function
returns fixed in-memory data. The docstrings and type hints are load-bearing, not
decoration - the LLM reads them to decide which tool to call and how to fill in arguments,
so a vague docstring produces worse tool selection, the same way a vague RAG chunk produces
worse retrieval.

Swapping these bodies for real API/database calls later doesn't change anything else in the
system - the router and the LLM's decision-making are entirely decoupled from how a tool is
implemented internally.
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
    """Search available flights between two cities on a given date.

    Args:
        origin: departure city or airport, e.g. 'Paris'
        destination: arrival city or airport, e.g. 'Algiers'
        departure_date: date of departure, e.g. '2026-08-13' or 'tomorrow'
    """
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
    """Get the current status of a specific flight: status, departure/arrival times, terminal, gate.

    Args:
        flight_number: the flight number, e.g. 'AH1235'
        date: the date of the flight, e.g. '2026-08-12'
    """
    return _FLIGHTS_DB.get(flight_number, {"error": f"No data found for flight {flight_number}"})


def get_airport_info(airport_code: str) -> dict:
    """Get information about an airport: name, city, terminals, timezone.

    Args:
        airport_code: IATA airport code, e.g. 'CDG'
    """
    return _AIRPORTS_DB.get(airport_code, {"error": f"No data found for airport {airport_code}"})


def get_booking(booking_reference: str) -> dict:
    """Get booking details by reference number: flight, date, passengers, class, baggage.

    Args:
        booking_reference: the booking reference code, e.g. 'ABC123'
    """
    return _BOOKINGS_DB.get(booking_reference, {"error": f"No booking found for reference {booking_reference}"})


TOOL_FUNCTIONS = {
    "search_flights": search_flights,
    "get_flight_status": get_flight_status,
    "get_airport_info": get_airport_info,
    "get_booking": get_booking,
}
