"""
Generate a realistic synthetic candidate dataset for development/demo.
Run once: python data/generate_dataset.py
Outputs: data/candidates.csv
"""
import csv
import json
import random
from pathlib import Path

random.seed(42)

CANDIDATES = [
    # ── Strong keyword match ──────────────────────────────────────────────────
    {
        "id": "C001",
        "name": "Alex Chen",
        "title": "Senior Full-Stack Engineer",
        "years_exp": 7,
        "skills": "React, Next.js, TypeScript, Node.js, Express, PostgreSQL, AWS, Docker",
        "work_auth": "US Citizen",
        "summary": "7 years building consumer-facing web products. Shipped Next.js 14 app serving 500K MAU. Led migration from CRA to Next.js, cutting TTI by 40%. Strong PostgreSQL schema design; introduced connection pooling that halved DB load.",
        "experience": json.dumps([
            {"role": "Senior SWE", "company": "Stripe", "years": 3, "bullets": ["Built Next.js checkout flow serving $2B+ transactions/year", "Owned PostgreSQL schema for payments; reduced query latency by 35%"]},
            {"role": "Full-Stack Engineer", "company": "Vercel", "years": 2, "bullets": ["Core contributor to Next.js App Router docs and examples", "Shipped TypeScript-first component library used by 200+ enterprise customers"]},
            {"role": "Frontend Engineer", "company": "Airbnb", "years": 2, "bullets": ["React performance optimization; reduced bundle size by 30%"]}
        ]),
        "github": "github.com/alexchen — 1.2K followers, 40+ repos, 3 open-source Next.js plugins",
        "education": "BS Computer Science, UC Berkeley",
        "notes": ""
    },
    {
        "id": "C002",
        "name": "Priya Sharma",
        "title": "Staff Engineer",
        "years_exp": 10,
        "skills": "Vue.js, Nuxt, Python, FastAPI, Redis, Kubernetes, GCP",
        "work_auth": "US Citizen",
        "summary": "10 years across frontend and backend. Vue/Nuxt expert; led team of 8 engineers. Strong platform background — designed multi-region Kubernetes deployment serving 2M users. Python FastAPI for all microservices. No Next.js but deep React-equivalent knowledge.",
        "experience": json.dumps([
            {"role": "Staff Engineer", "company": "Shopify", "years": 4, "bullets": ["Architect of Storefront API — Vue + FastAPI stack, 2M RPS at peak", "Built shared component system adopted across 4 product teams"]},
            {"role": "Senior Engineer", "company": "Twilio", "years": 3, "bullets": ["Led Nuxt.js migration from legacy PHP monolith", "Designed Redis caching layer; reduced API p99 from 800ms to 120ms"]},
            {"role": "Engineer", "company": "Pivotal Labs", "years": 3, "bullets": ["Consulted on 12 product teams; Vue, React, Angular — platform-agnostic"]},
        ]),
        "github": "github.com/priyasharma — Nuxt plugin with 3K stars",
        "education": "MS Computer Science, Stanford",
        "notes": "PLANTED: equivalent stack (Vue=React, Nuxt=Next.js) — keyword-poor but high-capability"
    },
    # ── Career switcher with transferable skills ───────────────────────────────
    {
        "id": "C003",
        "name": "Marcus Williams",
        "title": "Software Engineer (ex-Data Scientist)",
        "years_exp": 5,
        "skills": "Python, React, TypeScript, D3.js, PostgreSQL, dbt, Airflow",
        "work_auth": "US Citizen",
        "summary": "Career switcher from data science to full-stack. Built and shipped internal analytics dashboard in React + TypeScript that replaced a $150K/yr Tableau contract. Strong Python. Learning Next.js (completed official tutorial + built side project). High trajectory candidate.",
        "experience": json.dumps([
            {"role": "Software Engineer", "company": "Palantir", "years": 2, "bullets": ["Rewrote data pipeline UI in React/TypeScript; 10x faster than predecessor", "Shipped real-time dashboard serving 200 analysts daily"]},
            {"role": "Data Scientist", "company": "Spotify", "years": 3, "bullets": ["Built Python ML pipelines (Airflow + dbt)", "Created React prototype for recommendation explainability that shipped to 1M users"]},
        ]),
        "github": "github.com/marcuswilliams — Next.js portfolio project with 50 stars",
        "education": "BS Statistics, Columbia",
        "notes": "PLANTED: high-trajectory switcher; Next.js learner with strong React/TS foundation"
    },
    # ── Senior generalist, thin keyword match ────────────────────────────────
    {
        "id": "C004",
        "name": "Sarah Park",
        "title": "Principal Engineer",
        "years_exp": 12,
        "skills": "Angular, RxJS, Java, Spring Boot, Oracle, Azure",
        "work_auth": "US Citizen",
        "summary": "12 years in enterprise software. Angular specialist; led 15-person team. Strong Java/Spring backend. Less direct experience with React/Next.js stack but has led cross-stack migrations before. Recent upskilling in React.",
        "experience": json.dumps([
            {"role": "Principal Engineer", "company": "Accenture", "years": 5, "bullets": ["Led digital transformation for Fortune 100 client; migrated Angular 8 → 17", "Managed team of 15 across 3 time zones"]},
            {"role": "Senior Engineer", "company": "JPMorgan Chase", "years": 4, "bullets": ["Angular + Spring Boot trading dashboard; $5B daily volume", "Led security hardening initiative; zero breaches in 4-year tenure"]},
            {"role": "Engineer", "company": "IBM", "years": 3, "bullets": ["Java EE enterprise applications for 50K-user internal tools"]}
        ]),
        "github": "",
        "education": "BS Computer Engineering, Georgia Tech",
        "notes": ""
    },
    # ── Junior but impressive ─────────────────────────────────────────────────
    {
        "id": "C005",
        "name": "Jordan Kim",
        "title": "Full-Stack Developer",
        "years_exp": 2,
        "skills": "React, Next.js, TypeScript, Node.js, MongoDB, Vercel, Tailwind",
        "work_auth": "US Citizen",
        "summary": "2 years experience but shipped 3 production apps. Built SaaS side-project with Next.js + Stripe that reached $5K MRR. Active open-source contributor. Strong fundamentals; faster learner than most 5-year engineers per manager review.",
        "experience": json.dumps([
            {"role": "Full-Stack Developer", "company": "YC Startup (Series A)", "years": 2, "bullets": ["Next.js + tRPC + Prisma; sole frontend engineer", "Shipped MVP in 6 weeks; product now has 2K paying users"]}
        ]),
        "github": "github.com/jordankim — 500 followers, Next.js template with 800 stars",
        "education": "BS Computer Science, UT Austin",
        "notes": ""
    },
    # ── Overqualified / different domain ─────────────────────────────────────
    {
        "id": "C006",
        "name": "Elena Rodriguez",
        "title": "Senior ML Engineer",
        "years_exp": 8,
        "skills": "Python, PyTorch, FastAPI, React (basic), SQL, Spark, MLflow",
        "work_auth": "US Citizen",
        "summary": "8 years in ML engineering. Primarily Python/PyTorch. Built production FastAPI serving ML models to 10M users. Limited React experience but has shipped basic CRUD UIs. Applying for web engineer role as a pivot.",
        "experience": json.dumps([
            {"role": "Senior ML Engineer", "company": "OpenAI", "years": 3, "bullets": ["FastAPI microservices for GPT-4 inference; 10M daily requests", "Basic React dashboard for model monitoring"]},
            {"role": "ML Engineer", "company": "DeepMind", "years": 3, "bullets": ["PyTorch training pipelines; 200B parameter models"]},
            {"role": "Data Engineer", "company": "Google", "years": 2, "bullets": ["Spark + BigQuery data pipelines"]}
        ]),
        "github": "github.com/elenarodriguez — PyTorch contrib",
        "education": "PhD Computer Science (ML), MIT",
        "notes": ""
    },
    # ── Strong backend, weak frontend ────────────────────────────────────────
    {
        "id": "C007",
        "name": "James O'Brien",
        "title": "Backend Engineer",
        "years_exp": 6,
        "skills": "Go, Rust, gRPC, Kubernetes, PostgreSQL, Redis, Kafka",
        "work_auth": "Work Permit (H1-B active, valid through 2027)",
        "summary": "6 years in distributed systems. No JavaScript frontend experience. Backend infrastructure specialist — owns reliability of systems serving 50M users. Strong systems thinker. Applying for full-stack role but acknowledges frontend gap.",
        "experience": json.dumps([
            {"role": "Senior Backend Engineer", "company": "Cloudflare", "years": 3, "bullets": ["Go + Rust services handling 30M rps", "Zero-downtime Kubernetes rollouts for global edge network"]},
            {"role": "Backend Engineer", "company": "Discord", "years": 3, "bullets": ["Kafka message broker for 500M messages/day", "PostgreSQL sharding strategy for user data"]}
        ]),
        "github": "github.com/jamesobrien — 2K followers, Rust HTTP library with 5K stars",
        "education": "BS Computer Science, Trinity College Dublin",
        "notes": ""
    },
    # ── Good match, medium seniority ─────────────────────────────────────────
    {
        "id": "C008",
        "name": "Aisha Johnson",
        "title": "Mid-Level Full-Stack Engineer",
        "years_exp": 4,
        "skills": "React, TypeScript, Next.js, Node.js, Express, MySQL, Tailwind, GCP",
        "work_auth": "US Citizen",
        "summary": "4 years building e-commerce and SaaS products. Solid Next.js App Router experience. Contributed to a production Next.js codebase with 100K LOC. Good testing discipline (90% coverage on owned services). Collaborative; led two junior engineers.",
        "experience": json.dumps([
            {"role": "Full-Stack Engineer", "company": "Shopify Partner Agency", "years": 2, "bullets": ["Next.js storefronts for 15 merchant clients; up to 50K daily sessions", "TypeScript + tRPC + Prisma; zero type errors in production runtime"]},
            {"role": "Frontend Engineer", "company": "FinTech startup", "years": 2, "bullets": ["React dashboard for financial analytics; real-time WebSocket updates", "Led migration to Next.js 14 App Router; 25% improvement in Core Web Vitals"]}
        ]),
        "github": "github.com/aishajohnson — regular contributor",
        "education": "BS Information Systems, Howard University",
        "notes": ""
    },
    # ── Equivalent stack, Svelte/SvelteKit ───────────────────────────────────
    {
        "id": "C009",
        "name": "Dmitri Volkov",
        "title": "Frontend Engineer",
        "years_exp": 5,
        "skills": "Svelte, SvelteKit, TypeScript, Deno, Postgres, Bun, Vite",
        "work_auth": "US Citizen (naturalized)",
        "summary": "5 years in web development with Svelte as primary framework. SvelteKit is architecturally identical to Next.js (file-based routing, SSR/SSG/ISR, server components). Built a SaaS with SvelteKit serving 80K users. Creator of a popular Svelte testing library (2K GitHub stars). Completed React course and built 2 React apps; comfortable with the mental model.",
        "experience": json.dumps([
            {"role": "Senior Frontend Engineer", "company": "Linear", "years": 3, "bullets": ["SvelteKit SSR app; architecture mirrors Next.js App Router exactly", "Built component library with Storybook + TypeScript; 100% test coverage"]},
            {"role": "Frontend Engineer", "company": "Supabase", "years": 2, "bullets": ["SvelteKit + Postgres; open-source dashboard used by 200K developers", "Optimized Vite build; 60% faster HMR"]}
        ]),
        "github": "github.com/dmitrivolkov — svelte-test-utils with 2K stars",
        "education": "BS Computer Science, University of Helsinki",
        "notes": "PLANTED: SvelteKit ≈ Next.js (same paradigm); keyword-poor vs Next.js but high capability"
    },
    # ── Solid match, strong communication ────────────────────────────────────
    {
        "id": "C010",
        "name": "Mei Lin",
        "title": "Senior Full-Stack Engineer",
        "years_exp": 6,
        "skills": "React, Next.js, GraphQL, Python, Django, PostgreSQL, AWS",
        "work_auth": "Green Card",
        "summary": "6 years across startups and mid-size companies. Strong React/Next.js and solid Django backend. Good communicator — tech lead who writes RFCs that engineers actually read. Built real-time collaborative features (WebSockets). Mentors 3 junior engineers.",
        "experience": json.dumps([
            {"role": "Senior Full-Stack Engineer", "company": "Notion (contractor)", "years": 2, "bullets": ["Next.js collaborative editor with WebSocket real-time sync", "GraphQL API layer; reduced client data-fetching complexity by 50%"]},
            {"role": "Full-Stack Engineer", "company": "Figma", "years": 2, "bullets": ["React component library shared across web and mobile", "Django REST API for asset management; 5M assets served daily"]},
            {"role": "Engineer", "company": "Brex", "years": 2, "bullets": ["Next.js spending dashboard; launched with 0 accessibility violations"]}
        ]),
        "github": "github.com/meilin",
        "education": "BS Computer Science, UCLA",
        "notes": ""
    },
    # ── Consulting background, breadth over depth ─────────────────────────────
    {
        "id": "C011",
        "name": "Robert Tanaka",
        "title": "Tech Lead / Solutions Architect",
        "years_exp": 9,
        "skills": "React, Angular, Vue, Node.js, AWS, Terraform, Docker, Python",
        "work_auth": "US Citizen",
        "summary": "9 years in consulting — breadth over depth. Comfortable in any stack, but no single deep Next.js project. Led architecture for 20+ web products. Strong on system design, cloud infrastructure. Wants to return to hands-on IC role.",
        "experience": json.dumps([
            {"role": "Tech Lead", "company": "McKinsey Digital", "years": 4, "bullets": ["Architected digital platforms for 5 Fortune 500 clients across React/Angular/Vue", "Led team of 12 engineers; delivered $30M digital transformation project on time"]},
            {"role": "Senior Engineer", "company": "ThoughtWorks", "years": 3, "bullets": ["React + Node.js for retail e-commerce platforms", "Agile coaching alongside engineering; XP practices"]},
            {"role": "Developer", "company": "Accenture", "years": 2, "bullets": ["Angular applications for banking clients"]}
        ]),
        "github": "github.com/roberttanaka",
        "education": "BS Software Engineering, University of Washington",
        "notes": ""
    },
    # ── Not authorized to work ─────────────────────────────────────────────────
    {
        "id": "C012",
        "name": "Chen Wei",
        "title": "Full-Stack Engineer",
        "years_exp": 5,
        "skills": "React, Next.js, TypeScript, Node.js, PostgreSQL",
        "work_auth": "Student Visa (OPT expired - not authorized to work)",
        "summary": "5 years experience. Strong Next.js. OPT expired; currently not authorized to work in US.",
        "experience": json.dumps([
            {"role": "Full-Stack Engineer", "company": "ByteDance (US)", "years": 3, "bullets": ["Next.js platform; 10M users"]},
            {"role": "Engineer", "company": "Meta", "years": 2, "bullets": ["React components for Facebook web"]}
        ]),
        "github": "github.com/chenwei",
        "education": "MS Computer Science, Carnegie Mellon",
        "notes": "Should fail work authorization hard gate"
    },
    # ── Weak overall match ─────────────────────────────────────────────────────
    {
        "id": "C013",
        "name": "Lisa Thompson",
        "title": "Junior Developer",
        "years_exp": 1,
        "skills": "HTML, CSS, JavaScript, Bootstrap, WordPress",
        "work_auth": "US Citizen",
        "summary": "1 year of freelance web development. Built WordPress sites and landing pages. Learning React. No production full-stack experience.",
        "experience": json.dumps([
            {"role": "Freelance Developer", "company": "Self-employed", "years": 1, "bullets": ["WordPress sites for small businesses", "Basic HTML/CSS/JS landing pages"]}
        ]),
        "github": "",
        "education": "Coding bootcamp graduate",
        "notes": ""
    },
    # ── Remix framework (Next.js competitor/equivalent) ───────────────────────
    {
        "id": "C014",
        "name": "Gabriela Souza",
        "title": "Senior React Engineer",
        "years_exp": 6,
        "skills": "React, Remix, TypeScript, Node.js, Prisma, SQLite, Tailwind, Cloudflare Workers",
        "work_auth": "TN Visa (NAFTA - valid)",
        "summary": "6 years React specialist. Deep expertise in Remix (React Router v7 = Remix = conceptual superset of Next.js routing model). Shipped high-performance e-commerce platform with Remix serving 1M sessions/day. Server components, streaming SSR, edge rendering — knows this space cold. Would transition to Next.js in a day.",
        "experience": json.dumps([
            {"role": "Senior React Engineer", "company": "Shopify", "years": 3, "bullets": ["Remix (Hydrogen) storefront serving 1M sessions/day", "Migrated pages to streaming SSR; LCP improved from 3.2s → 1.1s"]},
            {"role": "React Engineer", "company": "Remix.run", "years": 2, "bullets": ["Core contributor to Remix framework", "Built Remix tutorial used by 100K developers"]},
            {"role": "Frontend Engineer", "company": "Platzi", "years": 1, "bullets": ["React + GraphQL educational platform", "Localized for 5M Spanish-speaking users"]}
        ]),
        "github": "github.com/gabrielasouza — Remix contrib, 3K followers",
        "education": "BS Systems Engineering, UNAM Mexico",
        "notes": "PLANTED: Remix ≈ Next.js (same data loading patterns, SSR paradigm); high capability"
    },
    # ── Backend-only, claiming full-stack ────────────────────────────────────
    {
        "id": "C015",
        "name": "Kevin Patel",
        "title": "Full-Stack Engineer",
        "years_exp": 5,
        "skills": "Node.js, Express, MongoDB, React (limited), Docker, AWS Lambda",
        "work_auth": "US Citizen",
        "summary": "5 years mostly backend. Claims full-stack but React experience is 6-month old project; hasn't shipped Next.js. Strong Node/Express. Decent systems thinking.",
        "experience": json.dumps([
            {"role": "Backend Engineer", "company": "Twitch", "years": 3, "bullets": ["Node.js + WebSocket for chat infrastructure; 10M concurrent users", "MongoDB sharding; 500M chat messages/day"]},
            {"role": "Full-Stack Engineer", "company": "Startup", "years": 2, "bullets": ["Express API + basic React frontend", "Sole engineer; shipped MVP but UI quality was poor (per design review)"]}
        ]),
        "github": "github.com/kevinpatel",
        "education": "BS Computer Science, Penn State",
        "notes": ""
    },
    # ── Strong trajectory, recent Next.js adoption ────────────────────────────
    {
        "id": "C016",
        "name": "Fatima Al-Hassan",
        "title": "Software Engineer",
        "years_exp": 4,
        "skills": "React, Next.js, TypeScript, GraphQL, Python, Flask, Redis, Docker",
        "work_auth": "US Citizen",
        "summary": "4 years with clear upward trajectory. Started as Python/Flask backend engineer; self-taught React and Next.js in year 2; now leading frontend architecture decisions. First Next.js project shipped to 200K users. Strong TypeScript. Active in tech community — speaks at local meetups.",
        "experience": json.dumps([
            {"role": "Software Engineer", "company": "HubSpot", "years": 2, "bullets": ["Next.js 14 App Router for marketing platform; 200K daily users", "Led TypeScript migration; reduced runtime errors by 60%"]},
            {"role": "Backend Engineer", "company": "Salesforce (intern → FTE)", "years": 2, "bullets": ["Flask REST APIs; transitioned to full-stack in year 2", "GraphQL federation layer across 5 microservices"]}
        ]),
        "github": "github.com/fatimaalhassan — open source contributions to Next.js",
        "education": "BS Computer Science, Howard University",
        "notes": ""
    },
    # ── Ember.js veteran ─────────────────────────────────────────────────────
    {
        "id": "C017",
        "name": "Tom Bradley",
        "title": "Senior Frontend Engineer",
        "years_exp": 8,
        "skills": "Ember.js, Handlebars, Ruby on Rails, PostgreSQL, Heroku, Backbone",
        "work_auth": "US Citizen",
        "summary": "8 years in Ember.js — a framework with similar component model and SSR concepts to React/Next.js. Expert in conventions-over-configuration approach. No React production experience but understands component lifecycle, state management, data fetching patterns well. Slower to adopt modern stack.",
        "experience": json.dumps([
            {"role": "Senior Frontend Engineer", "company": "LinkedIn (Ember era)", "years": 4, "bullets": ["Ember.js premium subscription UI; 50M users", "Designed reusable component system for 20-engineer team"]},
            {"role": "Frontend Engineer", "company": "Groupon", "years": 4, "bullets": ["Ember + Rails combo; 10M daily deal viewers"]}
        ]),
        "github": "github.com/tombradley",
        "education": "BA Computer Science, Ohio State",
        "notes": ""
    },
    # ── Strong impact, some stack mismatch ───────────────────────────────────
    {
        "id": "C018",
        "name": "Nina Okafor",
        "title": "Senior Software Engineer",
        "years_exp": 7,
        "skills": "React, Redux, TypeScript, Python, Django REST, PostgreSQL, AWS S3, CircleCI",
        "work_auth": "US Citizen",
        "summary": "7 years. React/Redux expert (before hooks era through modern). No Next.js specifically, but deep React understanding — would adopt Next.js in a sprint. Python/Django backend. Led 3-person engineering team, increased deployment frequency from monthly to daily.",
        "experience": json.dumps([
            {"role": "Senior Software Engineer", "company": "DoorDash", "years": 3, "bullets": ["React SPA for merchant portal; 500K merchants onboarded", "Redux → Zustand migration; 20% bundle reduction", "Led engineering team; instituted CI/CD; deployment frequency 10x improvement"]},
            {"role": "Software Engineer", "company": "Lyft", "years": 2, "bullets": ["React driver-partner dashboard", "Django REST API for earnings calculation system"]},
            {"role": "Engineer", "company": "Yelp", "years": 2, "bullets": ["React + Python for business owner tools"]}
        ]),
        "github": "github.com/ninaokafor",
        "education": "BS Computer Science, Spelman College",
        "notes": ""
    },
    # ── Very junior Next.js ───────────────────────────────────────────────────
    {
        "id": "C019",
        "name": "Jake Morrison",
        "title": "Junior Developer",
        "years_exp": 1,
        "skills": "Next.js, React, JavaScript, CSS Modules, Vercel",
        "work_auth": "US Citizen",
        "summary": "1 year experience. Bootcamp grad who learned Next.js during course. Shipped 2 portfolio projects. No production experience. Enthusiastic and learning fast.",
        "experience": json.dumps([
            {"role": "Junior Developer", "company": "Local Agency", "years": 1, "bullets": ["Next.js marketing sites for local businesses", "No production load (<100 users)"]}
        ]),
        "github": "github.com/jakemorrison — portfolio projects",
        "education": "Full-stack bootcamp, App Academy",
        "notes": ""
    },
    # ── Strong Astro/Qwik experience ────────────────────────────────────────
    {
        "id": "C020",
        "name": "Yuki Tanaka",
        "title": "Performance Engineer",
        "years_exp": 5,
        "skills": "Astro, Qwik, TypeScript, Deno, Web Components, CSS, Bun",
        "work_auth": "O-1 Visa (extraordinary ability)",
        "summary": "5 years obsessed with web performance. Built with Astro (island architecture, SSG/SSR, exactly like Next.js static/dynamic), Qwik (resumability — superset of SSR concepts). Two websites I built scored 100 Lighthouse on mobile. Deep understanding of rendering paradigms that underpins all of Next.js. React component model fully understood.",
        "experience": json.dumps([
            {"role": "Performance Engineer", "company": "Deno Land", "years": 2, "bullets": ["Astro + Deno Deploy; <1s TTFB globally", "Wrote Astro integration used by 5K projects"]},
            {"role": "Frontend Engineer", "company": "Cloudflare Pages team", "years": 3, "bullets": ["Framework-agnostic SSR worker runtime supporting Next.js, Astro, Remix", "Deep understanding of Next.js internals from building the adapter"]}
        ]),
        "github": "github.com/yukitanaka — Next.js adapter, 1K stars",
        "education": "MS Computer Science, Keio University",
        "notes": "PLANTED: framework-agnostic expert; built Next.js adapter; high capability despite no 'Next.js projects'"
    },
]

def main():
    out = Path(__file__).parent / "candidates.csv"
    fields = ["id","name","title","years_exp","skills","work_auth","summary","experience","github","education","notes"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in CANDIDATES:
            w.writerow(c)
    print(f"Wrote {len(CANDIDATES)} candidates to {out}")
    print("\nColumns:", fields)
    print("\nSample row:")
    c = CANDIDATES[0]
    for k,v in c.items():
        print(f"  {k}: {str(v)[:80]}")

if __name__ == "__main__":
    main()
