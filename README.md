🚗 What is this project?

This project implements a Car Damage Explainer Agent that:

Takes a car damage photo (and an optional text description),

Uses Gemini Vision to analyze visible damage,

Wraps the result into structured JSON using custom Python tools, and

Generates a professional-looking Markdown report for a fictional insurer:
“SKA Insurance – Car Damage Assessment Report”

The agent doesn’t just say “there is damage” — it:

Describes what is damaged,

Estimates severity (low / medium / high),

Suggests repair actions and a rough cost category, and

Presents everything in neat tables suitable for reports or apps.

This was built as a capstone for the Kaggle x Google “5-Day Gen AI Intensive – Agents” course, Freestyle track.

✨ Key Features

Multimodal damage analysis
Uses Gemini Vision (gemini-2.5-flash) to inspect car images and return structured JSON:

{
  "visible_damage": "...",
  "approx_severity": "low/medium/high",
  "location_hint": "...",
  "notes": "..."
}


Custom function tools
Python helpers for:

analyze_damage_description() – parse user text,

analyze_damage_image() – call Gemini Vision,

estimate_severity() – merge signals into a severity level,

suggest_repair_plan() – recommend next steps + safety notes.

Insurance-style AI report
The main agent (LlmAgent) produces a Markdown report including:

Header card with owner, car number, and brand (SKA Insurance),

Summary section,

Damage Assessment table,

Recommended Action & Rough Cost table,

Disclaimer section.

Notebook-friendly UX

Display the car image at the top,

Print the structured JSON going into the agent (debug + transparency),

Then show the final Markdown report.

🧱 High-Level Architecture

Input layer

User provides:

Image path (e.g. /kaggle/input/car-dmage/car_damage_2.jpeg)

Optional text like: “rear side, low-speed parking accident”

Optional metadata: owner name, car number, insurance brand.

Tool layer (Python functions)

analyze_damage_image()
→ Calls Gemini Vision with a strict JSON-only prompt and parses the response robustly (handles code fences, extra text, etc.).

analyze_damage_description()
→ Wraps human description into a small structured dict.

estimate_severity()
→ Uses Vision’s approx_severity and builds a reasoning string.

suggest_repair_plan()
→ Provides a generic repair path + safety note.

Agent layer (ADK LlmAgent)

car_damage_agent

Model: Gemini(model="gemini-2.5-flash", retry_options=retry_config)

Instruction: “You are a helpful Car Damage Explainer Agent…”

Receives a JSON blob (damage_info) in the prompt, not raw images.

Produces a Markdown report following a fixed layout.

Runner & helper

InMemoryRunner(agent=car_damage_agent)

run_car_damage_agent_with_image_and_text(...)

Orchestrates tools → builds damage_info → builds prompt → runs agent.

pretty_print_agent_response(response)

Cleans up the ADK response and prints nice readable text.

📁 Notebook Structure (Cell by Cell)
1. Environment & basic libraries
import os, json, base64, re
import numpy as np, pandas as pd
from PIL import Image

# List all files under /kaggle/input
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


Checks what data is available in /kaggle/input.

Confirms that your car damage images dataset is attached and shows full paths like:

/kaggle/input/car-dmage/car_damage_1.jpeg, etc.

2. ADK & Gemini imports
from kaggle_secrets import UserSecretsClient

from google.genai import types
from google import genai  # Vision

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService


Brings in:

Kaggle’s Secrets manager,

Gemini + ADK (LlmAgent, Runner, etc.).

3. Authentication & clients
# Get API key from Kaggle secret → GOOGLE_API_KEY
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Retry options for LLM calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Gemini Vision client
vision_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


Uses Kaggle Secrets → no API key hard-coded.

Configures retries for stability.

Creates a Vision client for multimodal image inspection.

4. Custom tools
analyze_damage_description(description: str)

Takes a free-text description from the user.

Returns a small structured dict with status, damage_type, location, notes.

analyze_damage_image(image_bytes: bytes, mime_type="image/jpeg")

The core Vision tool:

Encodes image as base64.

Sends it to gemini-2.5-flash with an instruction to respond ONLY with JSON.

Prints the raw text returned (🔍 Raw Vision model output) for debugging.

Tries:

Direct json.loads(),

Clean ```json code fences,

Regex to extract the first { ... } block.

Ensures the final dict contains:

visible_damage

approx_severity

location_hint

notes

status="success"

estimate_severity(damage_info: dict)

Pulls approx_severity from image_analysis.

Builds a human-readable reasoning string combining:

Vision’s description,

Vision notes,

User description.

Returns a dict:

{
  "status": "success",
  "severity_level": "low/medium/high",
  "reasoning": "..."
}

suggest_repair_plan(damage_info: dict)

Returns a high-level recommended repair path:

Visit body shop,

Professional difficulty,

Safety notes if sensors/lights may be affected.

All tools follow the ADK style: clear docstrings, type hints, status field.

5. Main agent definition: car_damage_agent
car_damage_agent = LlmAgent(
    name="car_damage_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""
        You are a helpful Car Damage Explainer Agent.
        ...
    """,
    tools=[],  # tools are called manually in Python
)

runner = InMemoryRunner(agent=car_damage_agent)


The agent does not call tools directly; instead, we pre-process with tools and pass the final JSON into its prompt.

The instruction tells it to:

Work only with provided JSON,

Not pretend to be a real mechanic,

Talk in clear language for car owners.

6. Orchestration + pretty printer
run_car_damage_agent_with_image_and_text(...)

This is the main function you use:

async def run_car_damage_agent_with_image_and_text(
    image_path: str,
    user_description: str = "",
    owner_name: str = "Shubham Ahire",
    car_number: str = "ABC-1234",
    insurance_company: str = "SKA Insurance"
):
    # Read the image
    # Call analyze_damage_image + analyze_damage_description
    # Build damage_info = { image_analysis, text_analysis }
    # Add severity_estimate = estimate_severity(damage_info)
    # Print damage_info for debugging
    # Build a Markdown-report prompt for the agent
    # Call: response = await runner.run_debug(prompt)
    # Return response


The prompt instructs the agent to generate a Markdown report with:

Header card including:

Insurance brand

Owner

Car No.

Type of assessment

### Summary section

### Damage Assessment table:

| Aspect          | Details |
|----------------|---------|
| Visible damage | ...     |
| Location       | ...     |
| Severity       | ...     |
| Notes          | ...     |


### Recommended Action & Rough Cost table:

| Item                  | Recommendation / Info |
|-----------------------|-----------------------|
| Suggested repair type | ...                   |
| Urgency               | ...                   |
| Rough cost category   | Low / Medium / High   |
| Safety notes          | ...                   |


### Disclaimer section.

pretty_print_agent_response(response)

ADK responses are nested objects; this helper:

Extracts content.text or content.parts[i].text,

Prints only the human-readable Markdown.

7. Final demo cell
image_path = "/kaggle/input/car-dmage/car_damage_2.jpeg"

print("📸 Car Image:")
if os.path.exists(image_path):
    display(Image.open(image_path))
else:
    print("❌ Image path not found:", image_path)

response = await run_car_damage_agent_with_image_and_text(
    image_path=image_path,
    user_description="This is a photo of the rear side of the car after a low-speed parking accident.",
    owner_name="Shubham Ahire",
    car_number="SKA-2025-09",
    insurance_company="SKA Insurance"
)

pretty_print_agent_response(response)


This cell:

Displays the selected car damage image at the top.

Runs the full pipeline:

Vision → tools → JSON → agent → Markdown report.

Prints the final report, which you can:

Screenshot for your Kaggle writeup,

Use as an example in the competition submission.

🛠 How to Run This Notebook (Kaggle)

Fork / Copy the notebook in Kaggle.

Click “Add data” and attach your car image dataset.

Confirm paths under /kaggle/input using the first cell.

Go to Add-ons → Secrets in the Kaggle notebook:

Create a secret named GOOGLE_API_KEY.

Paste your Gemini API key from Google AI Studio.

Run the cells from top to bottom, in order.

Update image_path, owner_name, car_number, etc. as needed.

Grab one of the Markdown reports + image for your:

Agents Intensive – Capstone submission,

GitHub README, or

Demo video.

⚠️ Limitations & Disclaimer

This is not a real insurance or repair system.

It does not inspect internal/structural damage — only visible external cues.

Cost and severity are very approximate and must not be used as legal or financial advice.

A real-world deployment would require:

Alignment with professional mechanics,

More robust cost modeling,

Proper validation and guardrails.

The notebook is meant as an educational demo of agentic architectures + multimodal Gemini in a domain aligned with traditional CV (car damage).

🚀 Possible Next Steps

Connect to a YOLO damage detector and feed its bounding boxes into the agent.

Log all cases and embed them into a vector store for similarity-based retrieval.

Wrap this notebook into a Streamlit / Gradio / web app to simulate a full SKA Insurance UI.

Extend language support (e.g., English + local languages) for wider accessibility.
