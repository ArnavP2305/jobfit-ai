
import html
import os

import streamlit as st
from dotenv import load_dotenv

from graph.workflow import analyze_resume
from utils.pdf_parser import extract_pdf_text


load_dotenv()

st.set_page_config(
    page_title="JobFit AI | Resume Intelligence",
    page_icon="✦",
    layout="wide",
)


def esc(value):
    return html.escape(str(value))


def report_text(result):
    lines = [
        "JOBFIT AI — RESUME MATCH REPORT",
        f"Overall weighted alignment: {result['match_score']}%",
        (
            f"Required score: {result['required_score']}%"
            if result["required_score"] is not None
            else "Required score: N/A"
        ),
        (
            f"Preferred score: {result['preferred_score']}%"
            if result["preferred_score"] is not None
            else "Preferred score: N/A"
        ),
        "",
        "SUMMARY",
        result.get("summary", ""),
        "",
        "REQUIREMENT EVIDENCE",
    ]

    for item in result["requirements"]:
        lines.extend([
            (
                f"- {item['skill']} | {item['priority']} | "
                f"{item['evidence_status']} | "
                f"{item['points_awarded']}/{item['points_possible']} points"
            ),
            (
                "  Resume evidence: "
                f"{item.get('resume_evidence') or 'No clear evidence found'}"
            ),
            f"  Job evidence: {item.get('jd_evidence') or ''}",
            f"  Rationale: {item.get('rationale') or ''}",
        ])

    lines.extend(["", "RECOMMENDATIONS"])

    lines.extend(
        f"- {item}"
        for item in result.get("recommendations", [])
    )

    lines.extend([
        "",
        "SUGGESTED SUMMARY",
        result.get("tailored_summary", ""),
        "",
        "DETAILED ANALYSIS",
        result.get("detailed_analysis", ""),
        "",
        "Scoring measures resume alignment, not hiring probability.",
    ])

    return "\n".join(lines)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(98,75,190,.22), transparent 35%),
        radial-gradient(ellipse at 90% 15%, rgba(31,181,150,.12), transparent 30%),
        #080b13;
    color: #f5f7ff;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.035em;
}

.hero {
    padding: 2rem 0 1.5rem;
}

.eyebrow {
    color: #8cf0d0;
    text-transform: uppercase;
    font-size: .75rem;
    letter-spacing: .18em;
    font-weight: 700;
}

.hero h1 {
    font-size: clamp(2.8rem, 6vw, 5rem);
    line-height: 1;
    margin: .7rem 0 1rem;
}

.gradient {
    background: linear-gradient(90deg, #8cf0d0, #a99bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub, .smallnote {
    color: #a6aec4;
}

.sub {
    font-size: 1.05rem;
    max-width: 700px;
    line-height: 1.7;
}

.panel {
    background: linear-gradient(145deg, rgba(22,29,46,.92), rgba(13,18,30,.92));
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 22px;
    padding: 1.35rem;
}

.kicker {
    color: #a6aec4;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .72rem;
    font-weight: 700;
}

.stTextArea textarea {
    background: #0c1220 !important;
    color: #f5f7ff !important;
    border: 1px solid #283248 !important;
    border-radius: 12px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #0c1220;
    border-color: #283248;
    border-radius: 10px;
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(100deg, #8cf0d0, #b5a8ff);
    color: #101523;
    font-weight: 800;
    border: 0;
    border-radius: 12px;
    padding: .75rem 1.1rem;
    min-height: 2.9rem;
}

.metric {
    background: #111827;
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;
    padding: 1.1rem;
    min-height: 110px;
}

.num {
    font: 700 2rem 'Space Grotesk';
    color: #8cf0d0;
}

.label {
    color: #a6aec4;
    font-size: .82rem;
    margin-top: .25rem;
}

.tag {
    display: inline-block;
    padding: .35rem .65rem;
    margin: .2rem;
    border-radius: 999px;
    font-size: .82rem;
    border: 1px solid rgba(255,255,255,.12);
}

.good {
    background: rgba(140,240,208,.09);
    color: #9ff5d9;
}

.partial {
    background: rgba(255,204,128,.09);
    color: #ffd59b;
}

.miss {
    background: rgba(255,143,143,.08);
    color: #ffb1b1;
}

hr {
    border-color: rgba(255,255,255,.1);
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ AI-powered career intelligence</div>
    <h1>Your next role.<br>
    <span class="gradient">A clearer match.</span></h1>
    <p class="sub">
        Compare your resume with a job description using weighted
        requirements and evidence from your actual resume.
    </p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### ✦ JobFit AI")

    model = st.selectbox(
        "Groq model",
        [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ],
        index=0,
    )

    st.caption(
        "Resume text and job description are sent to Groq for analysis."
    )

    st.markdown(
        "[Get a Groq API key](https://console.groq.com/keys)"
    )


left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="kicker">01 / Your profile</div>
        <h3>Bring your resume</h3>
        <p class="smallnote">Upload a PDF resume.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload resume (PDF)",
        type=["pdf"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="panel" style="margin-top:1rem">
        <div class="kicker">02 / Opportunity</div>
        <h3>Paste the job description</h3>
    </div>
    """, unsafe_allow_html=True)

    jd = st.text_area(
        "Job description",
        height=245,
        placeholder="Paste the complete job description...",
        label_visibility="collapsed",
    )


with right:
    st.markdown("""
    <div class="panel" style="height:100%">
        <div class="kicker">Inside your report</div>
        <h3>A score you can inspect.</h3>
        <p>Required requirements count twice as much as preferred requirements.</p>
        <p><b>Evidence levels</b><br>
        Supported = 100% · Partial = 50% · Not found = 0%</p>
        <p class="smallnote">
        This measures resume alignment, not your full ability or
        probability of getting hired.
        </p>
    </div>
    """, unsafe_allow_html=True)


if st.button("✦ Analyze my match", use_container_width=True):

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to your .env file.")

    elif not uploaded:
        st.warning("Upload your resume PDF first.")

    elif len(jd.strip()) < 80:
        st.warning(
            "Paste a more complete job description (at least 80 characters)."
        )

    else:
        try:
            with st.spinner(
                "Extracting requirements and checking evidence..."
            ):
                resume = extract_pdf_text(uploaded.getvalue())

                if len(resume) < 40:
                    raise ValueError(
                        "Very little text was extracted. "
                        "This may be a scanned PDF requiring OCR."
                    )

                result = analyze_resume(
                    resume,
                    jd.strip(),
                    api_key,
                    model,
                )

            st.markdown("---")

            st.markdown(
                '<div class="eyebrow">✦ Your analysis</div>'
                '<h2>Here is your role-fit snapshot.</h2>',
                unsafe_allow_html=True,
            )

            metrics = [
                (
                    f"{result['match_score']}%",
                    "Weighted resume alignment",
                ),
                (
                    (
                        f"{result['required_score']}%"
                        if result["required_score"] is not None
                        else "N/A"
                    ),
                    "Required skills score",
                ),
                (
                    (
                        f"{result['preferred_score']}%"
                        if result["preferred_score"] is not None
                        else "N/A"
                    ),
                    "Preferred skills score",
                ),
            ]

            columns = st.columns(3)

            for column, (number, label) in zip(columns, metrics):
                with column:
                    st.markdown(
                        f"""
                        <div class="metric">
                            <div class="num">{number}</div>
                            <div class="label">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.caption(
                "Scoring: required weight 2, preferred weight 1; "
                "supported evidence 1, partial 0.5, not found 0. "
                "This is not an ATS score."
            )

            st.markdown("### The short version")
            st.write(result.get("summary", ""))

            requirements = result["requirements"]

            groups = [
                ("✓ Evidence found", "supported", "good"),
                ("◐ Partial evidence", "partial", "partial"),
                ("◇ Not found in resume", "not_found", "miss"),
            ]

            columns = st.columns(3)

            for column, (title, status, css_class) in zip(
                columns, groups
            ):
                with column:
                    st.markdown(f"### {title}")

                    values = [
                        item["skill"]
                        for item in requirements
                        if item["evidence_status"] == status
                    ]

                    tags = "".join(
                        f'<span class="tag {css_class}">'
                        f'{esc(value)}</span>'
                        for value in values
                    )

                    st.markdown(
                        tags or '<span class="smallnote">None identified</span>',
                        unsafe_allow_html=True,
                    )

            st.markdown("### Requirement-by-requirement scoring")

            st.caption(
                "'Not found' means no evidence was found in the resume; "
                "it does not prove you lack the skill."
            )

            for item in requirements:
                with st.expander(
                    f"{item['skill']} — "
                    f"{item['evidence_status'].replace('_', ' ').title()}"
                ):
                    st.write(
                        f"**Priority:** {item['priority'].title()} "
                        f"(weight {item['points_possible']})"
                    )

                    st.write(
                        f"**Points:** {item['points_awarded']} / "
                        f"{item['points_possible']}"
                    )

                    st.write(
                        "**Resume evidence:** "
                        + (
                            item.get("resume_evidence")
                            or "No clear supporting passage found."
                        )
                    )

                    st.write(
                        "**Job requirement:** "
                        + (
                            item.get("jd_evidence")
                            or "Not specified"
                        )
                    )

                    if item.get("rationale"):
                        st.write(
                            f"**Reasoning:** {item['rationale']}"
                        )

            st.markdown("### Your action plan")

            for index, recommendation in enumerate(
                result.get("recommendations", []), 1
            ):
                st.markdown(f"**{index}.** {recommendation}")

            st.markdown("### Suggested professional summary")

            st.info(result.get("tailored_summary", ""))

            with st.expander("Detailed analysis"):
                st.write(result.get("detailed_analysis", ""))

            with st.expander("Extracted resume text"):
                st.text(resume[:15000])

            st.download_button(
                "Download detailed report",
                data=report_text(result),
                file_name="jobfit_analysis.txt",
                mime="text/plain",
            )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

            st.caption(
                "Check the Groq model ID, API key, connection, "
                "and whether the PDF contains selectable text."
            )


st.markdown("---")

st.markdown(
    '<p class="smallnote" style="text-align:center">'
    'JOBFIT AI · Transparent resume alignment, not a hiring guarantee.'
    '</p>',
    unsafe_allow_html=True,
)
