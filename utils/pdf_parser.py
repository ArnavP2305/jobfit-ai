
from io import BytesIO
import fitz


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF without saving it to disk."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with fitz.open(
            stream=BytesIO(pdf_bytes),
            filetype="pdf",
        ) as document:

            if document.is_encrypted:
                raise ValueError(
                    "This PDF is password-protected. "
                    "Upload an unlocked copy."
                )

            text = "\n\n".join(
                page.get_text("text")
                for page in document
            )

            return text.strip()

    except ValueError:
        raise

    except Exception as exc:
        raise ValueError(
            "Could not read this PDF. Please upload a valid PDF file."
        ) from exc
