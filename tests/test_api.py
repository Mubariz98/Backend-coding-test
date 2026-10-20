"""End-to-end HTTP tests with the real model. Each runs on Qdrant and on Chroma."""

from tests.pdf_factory import make_password_protected_pdf, make_pdf


def upload(client, name, data, content_type="application/pdf"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def query(client, **body):
    return client.post("/query", json=body)


# --- health ---------------------------------------------------------------

def test_health_reports_backend(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_db"] in ("qdrant", "chroma")


# --- upload ---------------------------------------------------------------

def test_upload_returns_summary(client):
    res = upload(client, "cats.pdf", make_pdf("Cats purr.", "Dogs bark."))
    assert res.status_code == 201
    body = res.json()
    assert (body["filename"], body["pages"], body["chunks"]) == ("cats.pdf", 2, 2)
    assert len(body["document_id"]) == 32


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
