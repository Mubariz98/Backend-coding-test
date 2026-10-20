"""End-to-end HTTP tests with the real embedding model. Each runs on Qdrant and on Chroma.

The LLM is replaced by FakeLLM where needed, so no Ollama is required.
"""

import json

from tests.fakes import FakeLLM
from tests.pdf_factory import make_password_protected_pdf, make_pdf


def upload(client, name, data, content_type="application/pdf"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def query(client, **body):
    return client.post("/query", json=body)


# --- health ---------------------------------------------------------------

def test_health_reports_backend_and_llm(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_db"] in ("qdrant", "chroma")
    assert body["llm_status"] == "unavailable"  # no Ollama in tests


def test_frontend_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


# --- upload ---------------------------------------------------------------

def test_upload_returns_summary(client):
    res = upload(client, "cats.pdf", make_pdf("Cats purr.", "Dogs bark."))
    assert res.status_code == 201
    body = res.json()
    assert (body["filename"], body["pages"], body["chunks"]) == ("cats.pdf", 2, 2)
    assert len(body["document_id"]) == 32
    assert body["size_bytes"] > 0 and body["uploaded_at"]


def test_duplicate_upload_returns_409(client):
    pdf = make_pdf("Identical content uploaded twice.")
    assert upload(client, "a.pdf", pdf).status_code == 201
    res = upload(client, "renamed.pdf", pdf)
    assert res.status_code == 409
    assert "already uploaded" in res.json()["detail"]


def test_rejects_non_pdf(client):
    assert upload(client, "notes.txt", b"hello", "text/plain").status_code == 400


def test_rejects_pdf_extension_with_wrong_content(client):
    assert upload(client, "fake.pdf", b"just text").status_code == 400


def test_rejects_empty_file(client):
    assert upload(client, "empty.pdf", b"").status_code == 400


def test_rejects_password_protected_pdf(client):
    res = upload(client, "locked.pdf", make_password_protected_pdf())
    assert res.status_code == 400
    assert "Password-protected" in res.json()["detail"]


def test_rejects_pdf_without_text(client):
    assert upload(client, "scan.pdf", make_pdf("")).status_code == 422


def test_rejects_oversized_file(make_client):
    client = make_client(max_upload_mb=1)
    res = upload(client, "big.pdf", b"%PDF-" + b"0" * (1024 * 1024))
    assert res.status_code == 413


def test_missing_file_field(client):
    assert client.post("/documents").status_code == 422


def test_filename_path_is_stripped(client):
    res = upload(client, "C:\\Users\\me\\report.pdf", make_pdf("Quarterly report."))
    assert res.json()["filename"] == "report.pdf"


# --- query ----------------------------------------------------------------

def test_query_on_empty_index_returns_no_results(client):
    res = query(client, query="anything")
    assert res.status_code == 200
    assert res.json()["results"] == []


def test_semantic_search_ranks_relevant_document_first(client):
    upload(client, "cats.pdf", make_pdf("Cats are small domestic felines that purr."))
    upload(client, "cars.pdf", make_pdf("The engine of a car burns fuel to move pistons."))

    top = query(client, query="pet animals", top_k=1).json()["results"][0]

    assert (top["filename"], top["page"]) == ("cats.pdf", 1)


def test_cross_lingual_query(client):
    upload(client, "garantie.pdf", make_pdf("Die Garantie fuer Neuwagen betraegt zwei Jahre."))
    upload(client, "kochen.pdf", make_pdf("Den Teig mit Zucker verruehren und backen."))

    top = query(client, query="How long is the car warranty?", top_k=1).json()["results"][0]

    assert top["filename"] == "garantie.pdf"


def test_filter_by_document_id(client):
    cats = upload(client, "cats.pdf", make_pdf("Cats purr softly.")).json()["document_id"]
    upload(client, "cars.pdf", make_pdf("Cars need fuel."))

    results = query(client, query="fuel for engines", document_id=cats).json()["results"]

    assert results and all(r["document_id"] == cats for r in results)


def test_min_score_filters_irrelevant_results(client):
    upload(client, "cars.pdf", make_pdf("The engine of a car burns fuel to move pistons."))
    res = query(client, query="best pancake recipe", min_score=0.9)
    assert res.json()["results"] == []


def test_top_k_limits_results(client):
    facts = ["Copper pipes resist corrosion.", "PVC pipes are cheap.", "Steel pipes are strong."]
    upload(client, "pipes.pdf", make_pdf(*facts))
    assert len(query(client, query="pipes", top_k=2).json()["results"]) == 2


def test_invalid_query_rejected(client):
    assert query(client, query="   ").status_code == 422
    assert query(client, query="ok", top_k=0).status_code == 422


# --- document management --------------------------------------------------

def test_list_get_and_delete_documents(client):
    first = upload(client, "first.pdf", make_pdf("Copper pipes resist corrosion.")).json()
    second = upload(client, "second.pdf", make_pdf("Cats purr softly.")).json()

    listed = client.get("/documents").json()
    assert [d["filename"] for d in listed] == ["second.pdf", "first.pdf"]  # newest first
    assert client.get(f"/documents/{first['document_id']}").json()["filename"] == "first.pdf"

    assert client.delete(f"/documents/{first['document_id']}").status_code == 204
    assert [d["filename"] for d in client.get("/documents").json()] == ["second.pdf"]
    # its chunks are gone from the vector DB too
    results = query(client, query="copper pipes").json()["results"]
    assert all(r["document_id"] == second["document_id"] for r in results)


def test_unknown_document_returns_404(client):
    assert client.get("/documents/nope").status_code == 404
    assert client.delete("/documents/nope").status_code == 404


def test_reupload_after_delete(client):
    pdf = make_pdf("Upload, delete, upload again.")
    doc_id = upload(client, "a.pdf", pdf).json()["document_id"]
    client.delete(f"/documents/{doc_id}")
    assert upload(client, "a.pdf", pdf).status_code == 201


# --- answers (LLM) --------------------------------------------------------

def test_ask_returns_answer_with_sources(make_client):
    llm = FakeLLM()
    client = make_client(llm=llm)
    upload(client, "garantie.pdf", make_pdf("Die Garantie fuer Neuwagen betraegt zwei Jahre."))

    body = client.post("/ask", json={"question": "Wie lange ist die Garantie?"}).json()

    assert body["answer"] == FakeLLM.ANSWER
    assert body["sources"][0]["filename"] == "garantie.pdf"
    assert "Die Garantie fuer Neuwagen" in llm.prompts[0]  # retrieved text reached the LLM


def test_ask_without_relevant_documents_does_not_call_llm(make_client):
    llm = FakeLLM()
    client = make_client(llm=llm)
    body = client.post("/ask", json={"question": "Anything?"}).json()  # empty index
    assert body["sources"] == [] and llm.prompts == []


def test_ask_stream_sends_ndjson_events(make_client):
    client = make_client(llm=FakeLLM())
    upload(client, "garantie.pdf", make_pdf("Die Garantie fuer Neuwagen betraegt zwei Jahre."))

    res = client.post("/ask/stream", json={"question": "Wie lange ist die Garantie?"})
    events = [json.loads(line) for line in res.text.splitlines()]

    assert res.headers["content-type"].startswith("application/x-ndjson")
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["filename"] == "garantie.pdf"
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == FakeLLM.ANSWER
    assert events[-1] == {"type": "done"}


def test_ask_returns_503_when_llm_is_down(client):
    upload(client, "garantie.pdf", make_pdf("Die Garantie fuer Neuwagen betraegt zwei Jahre."))
    question = {"question": "Wie lange ist die Garantie?"}
    assert client.post("/ask", json=question).status_code == 503
    assert client.post("/ask/stream", json=question).status_code == 503
    # search keeps working without the LLM
    assert query(client, query="Garantie").status_code == 200


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "  "}).status_code == 422
