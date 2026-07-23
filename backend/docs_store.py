"""
Document store for the brand knowledge base -- lets a non-technical user
upload, paste, and delete documents through the UI instead of needing to
edit files in the repo. Backs the same data/brand_docs/ directory that
rag.ingest_directory() already reads from, so no change to the RAG pipeline
itself -- only the entry point for getting content in changed.

Security-minded by necessity, since this is now a real upload surface:
- extension whitelist (.md, .txt, .pdf -- no executables, no arbitrary files)
- filename sanitization (no path separators, no '..', alnum/dash/underscore
  only) to prevent writing outside the docs directory
- size cap per file, to prevent someone uploading something enormous

PDF handling: PDFs are never stored as-is. Text is extracted at upload time
(pdfplumber) and saved as a .md file instead -- rag.py's ingestion only
understands text, and extracting once at upload avoids re-parsing the PDF on
every re-index. If extraction yields no usable text (e.g. a scanned/image-only
PDF with no OCR), the upload is rejected with a clear error rather than
silently indexing an empty document.
"""
import io
import os
import re
import uuid

import pdfplumber

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "brand_docs")
UPLOAD_ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}  # what a user may upload
STORED_EXTENSIONS = {".md", ".txt"}  # what actually ends up on disk (PDFs -> .md)
MAX_FILE_SIZE_BYTES = 2_000_000  # 2MB -- PDFs run larger than plain text docs
MAX_EXTRACTED_CHARS = 400_000  # sanity cap on extracted text, independent of file size

os.makedirs(DOCS_DIR, exist_ok=True)


class DocumentError(Exception):
    """Raised for any invalid upload/filename -- caught in main.py and
    returned as a 400, not a 500, since these are user input problems."""
    pass


def _sanitize_filename(name: str, allowed_extensions=UPLOAD_ALLOWED_EXTENSIONS) -> str:
    """Strip to a safe basename: letters, numbers, dash, underscore, one dot
    for the extension. Rejects path separators and '..' outright rather than
    trying to cleverly strip them, since silent stripping is how path
    traversal bugs slip through review."""
    if "/" in name or "\\" in name or ".." in name:
        raise DocumentError("Filename can't contain path separators or '..'")

    base, ext = os.path.splitext(name)
    ext = ext.lower()
    if ext not in allowed_extensions:
        raise DocumentError(f"Only {', '.join(sorted(allowed_extensions))} files are allowed")

    safe_base = re.sub(r"[^a-zA-Z0-9_\-]", "_", base).strip("_")
    if not safe_base:
        safe_base = f"doc_{uuid.uuid4().hex[:8]}"
    return safe_base + ext


def _safe_path(filename: str) -> str:
    """Resolve to an absolute path and verify it's actually inside DOCS_DIR --
    the real guard against path traversal, independent of filename sanitizing
    (defense in depth: even if sanitizing had a bug, this still catches it)."""
    docs_dir_abs = os.path.abspath(DOCS_DIR)
    candidate = os.path.abspath(os.path.join(docs_dir_abs, filename))
    if not candidate.startswith(docs_dir_abs + os.sep) and candidate != docs_dir_abs:
        raise DocumentError("Invalid filename")
    return candidate


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from PDF bytes via pdfplumber. Raises DocumentError if
    the PDF is unreadable or has no extractable text (e.g. a scanned image
    with no text layer -- this project doesn't run OCR)."""
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        raise DocumentError(f"Couldn't read PDF: {e}")

    text = "\n\n".join(text_parts).strip()
    if not text:
        raise DocumentError(
            "No extractable text found in this PDF -- it may be a scanned "
            "image without a text layer, which isn't supported here."
        )
    return text[:MAX_EXTRACTED_CHARS]


def list_documents():
    docs = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        if not fname.lower().endswith(tuple(STORED_EXTENSIONS)):
            continue
        path = os.path.join(DOCS_DIR, fname)
        stat = os.stat(path)
        docs.append({
            "name": fname,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        })
    return docs


def save_uploaded_bytes(filename: str, content: bytes) -> str:
    """Used for real file uploads (multipart). Returns the final saved
    filename (may differ from input if sanitized, de-duplicated, or -- for
    PDFs -- converted to .md after text extraction)."""
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise DocumentError(f"File too large -- max {MAX_FILE_SIZE_BYTES // 1_000_000}MB")
    if len(content) == 0:
        raise DocumentError("File is empty")

    safe_name = _sanitize_filename(filename)
    base, ext = os.path.splitext(safe_name)

    if ext == ".pdf":
        text = _extract_pdf_text(content)
        safe_name = base + ".md"
        content = text.encode("utf-8")
    elif ext not in STORED_EXTENSIONS:
        # shouldn't happen given _sanitize_filename's whitelist, but don't
        # silently write an unexpected extension to disk
        raise DocumentError(f"Unsupported file type: {ext}")

    path = _safe_path(safe_name)

    # avoid silently overwriting an existing doc with the same name
    if os.path.exists(path):
        base, ext = os.path.splitext(safe_name)
        safe_name = f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        path = _safe_path(safe_name)

    with open(path, "wb") as f:
        f.write(content)
    return safe_name


def save_text_document(name: str, text: str) -> str:
    """Used for the paste-text flow -- same validation path as file upload,
    just from a string instead of multipart bytes."""
    if not name.strip():
        raise DocumentError("Document name can't be empty")
    if not text.strip():
        raise DocumentError("Document content can't be empty")
    if not name.lower().endswith((".md", ".txt")):
        name = name + ".md"
    return save_uploaded_bytes(name, text.encode("utf-8"))


def delete_document(filename: str):
    safe_name = _sanitize_filename(filename, allowed_extensions=STORED_EXTENSIONS)
    path = _safe_path(safe_name)
    if not os.path.isfile(path):
        raise DocumentError(f"No such document: {filename}")
    os.remove(path)
