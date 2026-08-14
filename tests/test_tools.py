"""Unit tests for the mocked tools: deterministic, no API key needed."""

from app.tools import get_airport_info, get_booking, get_flight_status, search_flights


def test_get_flight_status_known_flight():
    result = get_flight_status("AH1235", "2026-08-12")
    assert result["status"] == "cancelled"


def test_get_flight_status_unknown_flight_returns_error():
    result = get_flight_status("ZZ0000", "2026-08-12")
    assert "error" in result


def test_get_airport_info_known_airport():
    result = get_airport_info("CDG")
    assert result["city"] == "Paris"
    assert result["timezone"] == "Europe/Paris"


def test_get_airport_info_unknown_airport_returns_error():
    result = get_airport_info("ZZZ")
    assert "error" in result


def test_get_booking_known_reference():
    result = get_booking("ABC123")
    assert result["flight_number"] == "AH1235"
    assert result["destination"] == "Algiers"


def test_get_booking_unknown_reference_returns_error():
    result = get_booking("XXXXXX")
    assert "error" in result


def test_search_flights_echoes_requested_route():
    result = search_flights("Paris", "Algiers", "tomorrow")
    flight = result["flights"][0]
    assert flight["origin"] == "Paris"
    assert flight["destination"] == "Algiers"
