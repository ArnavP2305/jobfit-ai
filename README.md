
# JobFit AI — Weighted Resume Scoring

## Features

- PDF resume extraction using PyMuPDF.
- Job requirement extraction using a Groq-hosted LLM.
- Evidence-based skill classification.
- Deterministic weighted scoring using Python.
- Separate required and preferred skill scores.
- Requirement-by-requirement evidence report.
- Personalized resume improvement recommendations.
- Downloadable text report.

## Scoring formula

For each requirement:

points_awarded = priority_weight * evidence_value

Overall score:

round(100 * sum(points_awarded) / sum(priority_weights))

### Priority weights

- Required requirement: 2
- Preferred requirement: 1

### Evidence values

- Supported: 1.0
- Partial: 0.5
- Not found: 0.0

Required and preferred scores are also calculated separately.

"Not found" means that the resume did not clearly demonstrate
the requirement. It does not prove the candidate lacks the skill.

The score measures resume alignment. It is not an ATS score,
an objective measure of ability, or a hiring prediction.

## Setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Create a .env file containing:

   GROQ_API_KEY=your_real_key

5. Run the application:

   streamlit run app.py

The model selector starts with openai/gpt-oss-120b.
If it is unavailable to your Groq account, select another
currently supported model in the sidebar.

## Limitations

The LLM still extracts job requirements and evaluates resume
evidence, so these classifications can be incorrect.

Check the evidence and recommendations before changing your resume.

The application sends extracted resume text and the job
description to Groq for analysis.

## Live Demo
https://jobfit-ai-00.streamlit.app/
