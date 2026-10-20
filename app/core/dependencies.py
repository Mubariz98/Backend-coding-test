"""FastAPI dependencies: hand routers the services built at startup.

Routers declare what they need with these `Annotated` aliases and never
construct anything themselves.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.core.container import Container
from app.documents.service import IngestionService
from app.search.service import SearchService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_settings(container: Annotated[Container, Depends(get_container)]) -> Settings:
    return container.settings


def get_ingestion_service(container: Annotated[Container, Depends(get_container)]) -> IngestionService:
    return container.ingestion


def get_search_service(container: Annotated[Container, Depends(get_container)]) -> SearchService:
    return container.search


SettingsDep = Annotated[Settings, Depends(get_settings)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
