# Summit Inbox Agent

A Python AI inbox agent that triages and drafts replies to customer email for Summit Roofing.

It reads a mailbox, classifies each email, extracts the customer's details,
lists what is still missing, and saves a reply as a Gmail draft for a human to
approve. It never sends anything to a customer.

## Configuration

**A new business adopts this agent by editing one file: `config/company.yaml`.**
Nothing company-specific lives in the prompt, the code, the sample emails, or
the evaluation set. The facts in that file are injected into the system prompt,
and the agent answers customer questions from them and nothing else — if a fact
is not in the file, the draft will not assert it.

| Key | What it does |
| --- | --- |
| `name`, `tagline` | Named in the prompt and used to build the sign-off, e.g. "The Summit Roofing Team" |
| `language` | The company's own language, as an ISO code |
| `reply_language` | `match_customer` (default) replies in whatever language the customer wrote in; `company` always uses `language`; or give a code such as `de` to fix it |
| `currency` | For any figure the business ever states. The agent never quotes prices |
| `contact.phone`, `contact.emergency_phone` | Quoted in drafts. The emergency number is given to every emergency |
| `services`, `not_offered` | What to offer, and what to decline plainly rather than offer to arrange |
| `service_area` | **Optional.** Leave it empty (`{}`) to serve customers anywhere — the agent then never declines on location grounds and never mentions geography. Set a `description` and optional `areas` list to restrict coverage. An emergency is never declined for location either way |
| `hours`, `response_times` | Answered directly when a customer asks. Response times such as "within 24 hours" are the only time commitments a draft may make |
| `warranty`, `payment` | Guardrails. Both tell the agent what it must *not* state |

Changing any value in the file changes the prompt fingerprint, which retires
every cached analysis automatically — so edits take effect on the next run
rather than being masked by stale cached results.

Two environment files sit alongside it: `.env` holds `ANTHROPIC_API_KEY` and the
optional `ANTHROPIC_MODEL` (see `.env.example`), and `credentials.json` /
`token.json` hold the Gmail OAuth client and token. All three are gitignored.

## What is not yet universal

The category set and the urgency rubric are roofing-shaped: "emergency" is
defined in terms of water entering a building and exposed roof structure. A
business in another trade would need those definitions changed in
`src/classifier.py`. Everything else — identity, contact details, services,
coverage, hours, language — comes from the config file.

## Running it

```bash
python demo_run.py            # triage the bundled sample emails
python evals/run_eval.py      # score the agent against evals/cases.json
streamlit run app.py          # review drafts in the dashboard
python -m src.gmail_client    # check the Gmail OAuth setup
```
