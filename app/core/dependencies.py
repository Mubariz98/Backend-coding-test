"""FastAPI dependencies: hand routers the services built at startup.

Routers declare what they need with these `Annotated` aliases and never
construct anything themselves.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.answers.ports import LLM
from app.answers.service import AnswerService
from app.core.config import Settings
from app.core.container import Container
from app.documents.service import DocumentService
from app.search.service import SearchService


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    return container.settings


def get_document_service(container: ContainerDep) -> DocumentService:
    return container.documents


def get_search_service(container: ContainerDep) -> SearchService:
    return container.search


def get_answer_service(container: ContainerDep) -> AnswerService:
    return container.answers


def get_llm(container: ContainerDep) -> LLM:
    return container.llm


SettingsDep = Annotated[Settings, Depends(get_settings)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
AnswerServiceDep = Annotated[AnswerService, Depends(get_answer_service)]
LLMDep = Annotated[LLM, Depends(get_llm)]
