from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class LlmConfig(BaseModel):
	enabled: bool = True
	api_key: Optional[str] = None
	base_url: Optional[str] = None
	model: str = "gpt-4o-mini"
	temperature: float = 0.0
	max_tokens: int = 600


class LinkCandidate(BaseModel):
	url: str
	text: Optional[str] = None
	context: Optional[str] = None
	source_url: str
	filtered_by: Literal["regex", "llm", "both", "manual"] = "regex"
	score: float = 0.0
	reason: Optional[str] = None


class ContactInfo(BaseModel):
	emails: List[str] = Field(default_factory=list)
	phones: List[str] = Field(default_factory=list)
	website: Optional[str] = None
	address: Optional[str] = None


class ProfileResult(BaseModel):
	university_name: Optional[str] = None
	program_name: Optional[str] = None
	professor_name: Optional[str] = None
	title: Optional[str] = None
	contact_info: ContactInfo = Field(default_factory=ContactInfo)
	research_interest: Optional[str] = None
	research_interest_summary: Optional[str] = None
	research_interest_chinese: Optional[str] = None
	keywords: List[str] = Field(default_factory=list)
	profile_url: str
	is_professor_page: bool = False
	is_accepting_phd: Optional[bool] = None  # True/False/None for unknown
	source: Literal["regex", "heuristic", "llm", "mixed"] = "heuristic"


class CrawlStats(BaseModel):
	directory_urls: List[str] = Field(default_factory=list)
	candidates_found: int = 0
	profiles_processed: int = 0
	llm_calls: int = 0
