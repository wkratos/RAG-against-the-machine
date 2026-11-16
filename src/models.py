from typing import List
import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """The MinimalSource model represents a single source of information"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """The UnansweredQuestion model represent an unanswered
    question"""
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """The AnsweredQuestion model represent an answered question"""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """The RagDataset model represents a dataset of RAG questions"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The MinimalSearchResults model represent the search results"""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The MinimalAnswer model represent an answer"""
    answer: str


class StudentSearchResults(BaseModel):
    """The StudentSearchResults model represent search results"""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """The StudentSearchResultsAndAnswer model represent search results
    with answers"""
    search_results: List[MinimalAnswer]
    k: int
