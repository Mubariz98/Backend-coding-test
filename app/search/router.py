"""HTTP endpoints for querying documents."""

from fastapi import APIRouter

from app.core.dependencies import SearchServiceDep
from app.search.schemas import QueryRequest, QueryResponse, SearchResult

router = APIRouter(prefix="/query", tags=["search"])


# Plain `def`: FastAPI runs it in a threadpool, so embedding the query doesn't block the event loop.
@router.post("")
def query_documents(request: QueryRequest, service: SearchServiceDep) -> QueryResponse:
    hits = service.search(request.query, request.top_k, request.document_id, request.min_score)
    return QueryResponse(query=request.query, results=[SearchResult.from_hit(h) for h in hits])
