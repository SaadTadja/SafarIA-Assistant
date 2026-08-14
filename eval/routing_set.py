"""31 routing scenarios, for a resolution of 3.2% per case instead of the 16.7% that the
brief's 6 allow. Those 6 are still reported separately in run_eval.py.

expected_source follows run_eval.py: tool names joined by '+', 'rag' for
search_knowledge_base, 'llm' when nothing should be called.
"""

# (query, expected_source)
ROUTING_SCENARIOS = [
    # --- search_flights (5) ---
    ("Find me a flight from Paris to Algiers tomorrow.", "search_flights"),
    ("I need to fly from Casablanca to Paris next Monday.", "search_flights"),
    ("What flights are available from Algiers to Casablanca on 20 August?", "search_flights"),
    ("Cherche un vol de Paris a Casablanca le 15 septembre.", "search_flights"),
    ("Show me flights Paris to Algiers on 2026-09-01.", "search_flights"),

    # --- get_flight_status (5) ---
    ("What is the status of flight AH1235?", "get_flight_status"),
    ("Is AH1009 on time?", "get_flight_status"),
    ("Quel est le statut du vol AH1235 ?", "get_flight_status"),
    ("Which gate does AH1009 leave from?", "get_flight_status"),
    ("Has flight AH1235 been delayed or cancelled?", "get_flight_status"),

    # --- get_booking (4) ---
    ("Give me the information for booking ABC123.", "get_booking"),
    ("What class is booking ABC123 in?", "get_booking"),
    ("How many passengers are on reservation ABC123?", "get_booking"),
    ("Look up my reservation ABC123 please.", "get_booking"),

    # --- get_airport_info (4) ---
    ("Give me information about CDG airport.", "get_airport_info"),
    ("What terminals does CMN have?", "get_airport_info"),
    ("What timezone is ALG airport in?", "get_airport_info"),
    ("Tell me about Casablanca CMN airport.", "get_airport_info"),

    # --- rag (7) --- the last names a flight while asking a policy question
    ("What are the cabin baggage rules?", "rag"),
    ("How much does excess baggage cost?", "rag"),
    ("Do I need a visa to travel to Europe?", "rag"),
    ("How early should I check in for an international flight?", "rag"),
    ("Comment voyager avec un animal de compagnie ?", "rag"),
    ("What is the refund policy if I cancel my own ticket?", "rag"),
    ("I'm flying on AH1009 - what's the checked baggage allowance?", "rag"),

    # --- hybrid: state first, then policy (3) ---
    ("My flight AH1235 is cancelled. Can I get a refund?", "get_flight_status+rag"),
    ("AH1235 got cancelled - what are my rebooking options?", "get_flight_status+rag"),
    ("Is AH1235 cancelled, and if so what compensation applies?", "get_flight_status+rag"),

    # --- llm only: nothing should fire (3) ---
    ("Hello, who are you?", "llm"),
    ("Thanks, that was helpful!", "llm"),
    ("Bonjour, que peux-tu faire ?", "llm"),
]

CATEGORY_OF = {
    "search_flights": "flight_search",
    "get_flight_status": "flight_status",
    "get_booking": "booking",
    "get_airport_info": "airport",
    "rag": "policy",
    "get_flight_status+rag": "hybrid",
    "llm": "conversational",
}
