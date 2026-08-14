# SafarIA Assistant

Un assistant de voyage basé sur un LLM, qui répond aux questions de politique tarifaire via
une base de connaissances RAG et aux questions propres à l'utilisateur via des outils, le
tout routé par une seule boucle de function calling. Construit sur FastAPI et le SDK OpenAI
directement, sans LangChain ni LlamaIndex. Inspiré de la charte graphique de Royal Air
Maroc ; « Safar » (voyage) + « IA ».

## Architecture

```
                    POST /chat  (ou /chat/stream)
                              |
                              v
              LLM via OpenRouter — system prompt + 5 outils
                              |
      +-----------+-----------+-----------+------------------+
      v           v           v           v                  v
search_flights  get_flight_  get_booking  get_airport_  search_knowledge_base
                status                    info          (encapsule le RAG)
      |           |           |           |                  |
      +-----------+-----------+-----------+------------------+
                              v
              Réponse, combinant éventuellement plusieurs outils
              {"answer": "...", "source": "get_flight_status+rag"}
```

**La décision de conception qui compte :** le RAG est exposé au modèle comme un *cinquième
outil*, et non comme une branche écrite à la main du type « si question de politique, alors
RAG, sinon outils ». Une seule boucle gère tout le routage, y compris les cas hybrides :
elle peut appeler `get_flight_status` pour confirmer une annulation, puis
`search_knowledge_base` pour la politique de remboursement, avant de répondre. Le modèle ne
voit jamais le texte des documents, uniquement le résultat d'un appel de récupération.

## Livrables

| Attendu | Où |
|---|---|
| API `/chat` | `app/main.py` — plus `/chat/stream` (SSE) et `/health` |
| Pipeline RAG | `app/rag.py` — chargement, découpage, normalisation des requêtes, récupération en deux étapes, seuil de confiance |
| Intégration des outils | `app/tools.py` (4 outils simulés) + `app/router.py` (schémas, boucle de function calling) |
| Tests | `tests/` — **42 tests passants** |
| Interface | `app/static/index.html` — sans étape de build |
| Évaluation | `eval/` — récupération, routage, juge LLM, robustesse |

## Lancer le projet

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
echo OPENROUTER_API_KEY=votre-cle-ici > .env
uvicorn app.main:app
```

Le modèle est `openai/gpt-4o-mini`, interchangeable via `MODEL_NAME` dans `router.py`.
Interface sur http://127.0.0.1:8000/ — chaque réponse porte un badge indiquant sa `source`.

```bash
pytest                           # 42 tests ; ceux en direct sont ignorés sans clé API
python -m eval.run_eval          # routage, juge, coût — nécessite une clé
python -m eval.robustness_eval   # récupération seule, aucune clé nécessaire
```

## Comment le choix RAG / Tools est fait

Le system prompt énonce la règle explicitement : les questions de politique et de procédure
vont vers `search_knowledge_base` ; les questions portant sur un vol, une réservation ou un
aéroport précis vont vers l'outil correspondant ; certaines questions nécessitent les deux,
et le modèle est instruit de vérifier d'abord l'état dynamique, puis la politique
applicable. Cela reproduit la façon dont un agent humain traite le problème, et c'est ce
qui permet au scénario « vol annulé → remboursement » de fonctionner sans aucune ligne de
routage écrite en dur.

## Comment les appels inutiles sont évités

Deux couches indépendantes, pas une seule.

1. **Les descriptions des outils sont précises et mutuellement exclusives.** Celle de
   `search_knowledge_base` précise ce à quoi elle ne sert *pas*, ce qui l'empêche de se
   déclencher sur une recherche de vol.
2. **Le seuil de confiance** dans `RagIndex.search` rejette la récupération avant qu'elle
   n'atteigne le LLM lorsque le meilleur score après reranking est inférieur à `0.4` : une
   question hors sujet ne coûte donc aucun appel de génération. Taux de rejet hors périmètre
   mesuré : **100 % sur 10 requêtes**.

Les échanges de politesse servent de test de non-régression : « Hello, who are you ? » doit
produire `source: "llm"` sans aucun appel d'outil. **Taux d'appels inutiles : 0 %.**

## Comment les hallucinations sont limitées

- Le prompt interdit d'inventer des détails, interdit d'étendre une politique à une
  situation qu'elle ne couvre pas, et impose d'indiquer clairement ce que l'assistant ne
  peut pas faire.
- Le seuil de confiance garantit qu'un contexte faible n'atteint jamais le modèle, ce qui
  supprime la tentation d'étirer un extrait à moitié pertinent pour en tirer une réponse.
- Le prompt interdit explicitement de répondre de mémoire sur les documents d'entrée — visa,
  passeport — même lorsque le modèle croit savoir. C'est une correction issue d'une mesure :
  ces questions posées en anglais ne consultaient jamais la base (0/6), contre 6/6 après.
- Chaque fait utilisé provient d'une sortie d'outil structurée, jamais d'un texte libre que
  le modèle aurait produit lui-même.
- **La fidélité est mesurée par un juge LLM, et non par correspondance de mots-clés**
  (`eval/judge.py`). Le juge est lui-même validé sur 5 cas étiquetés — dont une
  hallucination inventant une indemnisation et une réponse correctement négative — avant que
  ses chiffres ne soient utilisés : **5/5**.

## Métriques

Mesurées, non estimées. `openai/gpt-4o-mini` ; juge sur `claude-haiku-4.5`.

| Routage | |
|---|---|
| Précision de routage (31 scénarios) | **100 %** |
| Les 6 scénarios du brief | **100 %** |
| Les 12 exemples du brief, en français | **12/12** |
| Sélection d'outil / extraction des arguments | **100 %** / **100 %** |
| Appels d'outils inutiles sur small talk | **0 %** |
| Fuites par injection de prompt | **0** |

| Récupération | |
|---|---|
| Hit Rate@1 — questions bien formées (9) | **100 %** |
| Hit Rate@1 — formes de requêtes variées (26) | **92,3 %** |
| Passage du seuil de confiance (26) | **88,5 %** |
| Rejet hors périmètre (10) | **100 %** |

| Qualité des réponses (juge LLM) | 6 du brief | 11 cas limites |
|---|---|---|
| Fidélité moyenne | **1,000** | 0,932 |
| Réponses entièrement fidèles | **100 %** | 81,8 % |
| Pertinence de la réponse | **1,000** | 0,609 |
| Abstention injustifiée | **0 %** | 9,1 % |
| Validation du juge | **5/5** | — |

| Exploitation | |
|---|---|
| Coût par requête | **0,00038 $** |
| Latence | 2,6 – 3,2 s |
| Démarrage (embeddings en cache) | 10,3 s, contre 18,8 s à froid |
| Tests | **42/42** |

**Comment lire ces chiffres honnêtement :** relancer deux fois la même évaluation fait
varier les scores agrégés du juge jusqu'à 0,166. Toute différence inférieure à cela sur une
seule exécution est du bruit, pas un signal. C'est précisément pour cette raison que le jeu
de scénarios de routage est passé de 6 à 31 : à n=6, un seul scénario pesait 16,7 %.

## Bonus

| Élément | Statut |
|---|---|
| Mémoire conversationnelle | ✅ `session_id` sur `/chat` ; un test en direct vérifie que le modèle résout « it » à partir du tour précédent |
| Gestion correcte des erreurs API | ✅ Backoff sur 429/5xx, plafonné à 60 s ; 402 et rate limits traduits en erreurs HTTP propres ; 4 tests de non-régression |
| Streaming | ✅ `POST /chat/stream` (SSE) — événements `tool`, `token`, `done` ; l'interface lit de façon incrémentale |
| Observabilité des appels LLM | ✅ Une ligne JSON par tour (routage, outils, latence, tokens) + badge de source par réponse dans l'interface |
| Tests automatiques RAG / Tool | ✅ Les 6 scénarios du brief + questions de documents d'entrée + jeu de 31 scénarios de routage + non-régression sur la robustesse de la récupération |
| Interface utilisateur simple | ✅ Aux couleurs de la marque, bilingue EN/FR, rendu markdown, badges de source |

## Limites

- **La fidélité est évaluée par un LLM**, qui a son propre taux d'erreur. C'est une nette
  amélioration par rapport à la vérification par sous-chaîne qu'il remplace — celle-ci
  comptait une réponse défaillante comme réussie — mais ce n'est pas une vérité terrain.
- **Échantillons réduits.** 31 scénarios de routage, 26 requêtes de récupération, 10 hors
  périmètre. Ce sont de vraies mesures, avec de larges intervalles de confiance.
- **Les cas limites restent le point faible.** Sur les 11 scénarios difficiles, la pertinence
  des réponses tombe à 0,609 : l'assistant renvoie encore vers le service client plus souvent
  qu'il ne le devrait, et une abstention injustifiée subsiste. Les cas nominaux sont à 1,000.
- **Les requêtes courtes se récupèrent mal.** Une requête de deux mots obtient un score quasi
  nul avec le cross-encoder, même lorsque le classement est correct. Le routeur est instruit
  de formuler des questions complètes, ce qui évite le problème en pratique sans le résoudre.
- **Le grounding dépend du prompt, pas d'un garde-fou.** Les questions de visa en anglais
  recevaient une réponse tirée des connaissances du modèle plutôt que du corpus — 0/6 sur deux
  formulations, alors que la question française interrogeait bien la base. Corrigé par une
  règle explicite (6/6 après), mais rien n'empêche structurellement la même dérive sur un
  autre sujet : c'est une instruction, pas une contrainte.
- **Les outils sont simulés** (`app/tools.py`), et **l'historique de conversation est en
  mémoire** — acceptable pour une démo, inadapté à un déploiement, qui nécessiterait Redis ou
  une base de données avec TTL.
- **Pas de base vectorielle.** Similarité cosinus en force brute sur 312 chunks : suffisant à
  cette taille, ne passera pas l'échelle au-delà de quelques milliers.
