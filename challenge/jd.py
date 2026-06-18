"""
Job description text + JD-derived constants for scoring.
Single source of truth — both precompute and rank import from here.
"""

JD_TEXT = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities
Experience: 5–9 years

What you'd actually be doing:
Own the intelligence layer — ranking, retrieval, and matching systems.
Ship v2 ranking system using embeddings, hybrid retrieval, LLM-based re-ranking.
Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.
Drive candidate-JD matching architecture at scale.
Mentor next round of hires (team growing 4 → 12 engineers).

Things you absolutely need:
- Production experience with embeddings-based retrieval (sentence-transformers, OpenAI embeddings, BGE, E5)
- Production experience with vector databases / hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS, OpenSearch, Elasticsearch)
- Strong Python — code quality matters
- Hands-on experience designing evaluation frameworks for ranking (NDCG, MRR, MAP, offline-to-online correlation, A/B testing)

Nice to have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- HR-tech / recruiting-tech / marketplace products
- Distributed systems / large-scale inference optimization
- Open-source contributions in AI/ML

Explicit disqualifiers:
- Pure research without production deployment
- AI experience = only recent LangChain wrappers under 12 months
- Senior engineer who hasn't written code in 18+ months
- Exclusively consulting-firm career (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
- Pure CV/speech/robotics without NLP/IR exposure
- No visa sponsorship — must be India-based or willing to relocate to India

Ideal profile: 6-8 years, 4-5 in applied ML/AI at product companies.
Has shipped at least one end-to-end ranking/search/recommendation system at scale.
Located in or willing to relocate to Noida or Pune.
"""

# AI/ML skills from the JD — used for skill match scoring
# Order matters: earlier = more weight
TARGET_SKILLS = [
    # Must-have core
    "embeddings", "vector database", "vector search", "semantic search",
    "information retrieval", "retrieval", "ranking", "search",
    "sentence transformers", "sentence-transformers",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "meilisearch",
    "ndcg", "mrr", "map", "evaluation", "a/b testing",
    "hybrid search", "bm25", "dense retrieval", "sparse retrieval",
    # LLM / NLP
    "nlp", "natural language processing", "llm", "large language model",
    "transformers", "bert", "gpt", "rag", "retrieval augmented",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "openai", "anthropic", "hugging face", "huggingface",
    # ML Engineering
    "machine learning", "deep learning", "pytorch", "tensorflow",
    "recommendation system", "recommender", "learning to rank",
    "xgboost", "lightgbm", "gradient boosting",
    "mlops", "model serving", "inference optimization",
    # Engineering
    "python", "distributed systems", "kafka", "spark",
    "api", "microservices", "docker", "kubernetes",
]

# Large IT services firms — pure consulting career is a disqualifier
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl technologies",
    "hcl tech", "tech mahindra", "mphasis", "hexaware",
    "mindtree", "ltimindtree", "l&t infotech", "persistent systems",
    "niit technologies", "mastech", "zensar",
}

# India locations that match JD preference
PREFERRED_LOCATIONS = {
    "india", "pune", "noida", "hyderabad", "mumbai", "delhi",
    "delhi ncr", "ncr", "bangalore", "bengaluru", "chennai",
    "gurgaon", "gurugram", "navi mumbai", "thane",
}

# Proficiency → numeric score
PROFICIENCY_SCORE = {
    "beginner": 0.20,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}

# Education tier → numeric score
EDUCATION_TIER_SCORE = {
    "tier_1": 1.00,
    "tier_2": 0.75,
    "tier_3": 0.50,
    "tier_4": 0.25,
    "unknown": 0.40,
}

# Final score weights
WEIGHTS = {
    "semantic":    0.35,
    "skill_match": 0.28,
    "behavioral":  0.22,
    "experience":  0.10,
    "education":   0.05,
}

# Claude score replaces semantic + partial skill when available
CLAUDE_WEIGHT = 0.55   # how much pre-computed Claude score contributes
FEATURE_WEIGHT = 0.45  # how much rule-based features contribute when Claude score exists
