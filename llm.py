from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple
import json

from .models import LlmConfig, LinkCandidate

try:
	from openai import OpenAI
except Exception:  # pragma: no cover
	OpenAI = None  # type: ignore


class LlmClient:
	def __init__(self, cfg: LlmConfig) -> None:
		self.cfg = cfg
		openai_key = os.getenv("OPENAI_API_KEY")
		ark_key = os.getenv("ARK_API_KEY")
		api_key = cfg.api_key or openai_key or ark_key

		# base_url precedence: explicit cfg > OPENAI_BASE_URL > ARK_BASE_URL > default Ark if ARK key present > None
		base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("ARK_BASE_URL") or ("https://ark.cn-beijing.volces.com/api/v3" if ark_key else None)

		self.enabled = bool(cfg.enabled and api_key and OpenAI)
		self._client = None
		if self.enabled:
			self._client = OpenAI(api_key=api_key, base_url=base_url)

	def filter_professor_links(self, base_url: str, candidates: List[LinkCandidate], limit: int = 60) -> List[LinkCandidate]:
		if not self.enabled:
			return candidates
		short = candidates[:limit]
		items = [f"- url: {c.url}\n  text: {c.text or ''}\n  context: {c.context or ''}" for c in short]
		prompt = (
			"Given a list of links from a university site, keep only those that are likely individual professor profile pages (tenure-track or teaching faculty). Exclude students, staff, labs, groups, general directories, PDFs, news/events.\n"
			"Return a compact JSON object with key 'keep' as a list of objects: {url, reason}.\n"
			"Be conservative but not overly strict."
		)
		content = "\n".join(items)
		resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
			model=self.cfg.model,
			temperature=self.cfg.temperature,
			messages=[
				{"role": "system", "content": "You are a precise classifier for academic web pages."},
				{"role": "user", "content": f"{prompt}\n\nLinks from {base_url}:\n{content}\n\nReturn JSON only."},
			],
		)
		text = resp.choices[0].message.content or "{}"
		try:
			parsed = json.loads(text)
			keep_urls = {item.get("url") for item in (parsed.get("keep") or []) if item.get("url")}
			reasons = {item.get("url"): item.get("reason") for item in (parsed.get("keep") or []) if item.get("url")}
			filtered: List[LinkCandidate] = []
			for c in short:
				if c.url in keep_urls:
					c.filtered_by = "llm" if c.filtered_by == "regex" else "both"
					c.reason = reasons.get(c.url)
					filtered.append(c)
			return filtered if filtered else candidates
		except Exception:
			return candidates

	def analyze_profile(self, url: str, page_text: str) -> Dict[str, Any]:
		if not self.enabled:
			return {}
		instructions = (
			"You are extracting information from a professor's profile page. Determine if this is a professor's page, whether they accept PhD candidates, research interests, name, title, emails/phones, and keywords summarizing research.\n"
			"Return STRICT JSON with keys: is_professor_page (bool), professor_name (str|null), title (str|null), is_accepting_phd ('yes'|'no'|'unknown'), research_interest (str|null), keywords (str[]), emails (str[]), phones (str[])."
		)
		content = page_text[:12000]
		resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
			model=self.cfg.model,
			temperature=self.cfg.temperature,
			messages=[
				{"role": "system", "content": "You are a careful information extraction assistant for academic profiles."},
				{"role": "user", "content": f"URL: {url}\n\n{instructions}\n\nContent:\n{content}\n\nReturn JSON only."},
			],
		)
		text = resp.choices[0].message.content or "{}"
		try:
			return json.loads(text)
		except Exception:
			return {}


def _get_llm_client(cfg: LlmConfig) -> Optional[Any]:
	if not OpenAI or not cfg.enabled:
		return None
	
	openai_key = os.getenv("OPENAI_API_KEY")
	ark_key = os.getenv("ARK_API_KEY")
	api_key = cfg.api_key or openai_key or ark_key
	base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("ARK_BASE_URL") or ("https://ark.cn-beijing.volces.com/api/v3" if ark_key else None)
	
	if api_key and base_url:
		return OpenAI(api_key=api_key, base_url=base_url)
	return None


def translate_to_chinese(cfg: LlmConfig, text: str) -> Optional[str]:
	if not text:
		return None
	from .prompts import TRANSLATE_TO_CHINESE_PROMPT
	
	client = _get_llm_client(cfg)
	if not client:
		return None

	try:
		res = client.chat.completions.create(
			model=cfg.model,
			temperature=0.1,
			max_tokens=1024,
			messages=[
				{"role": "system", "content": TRANSLATE_TO_CHINESE_PROMPT},
				{"role": "user", "content": text},
			]
		)
		if res.choices:
			return res.choices[0].message.content.strip()
	except Exception as e:
		print(f"Error calling LLM for translation: {e}")
	return None


def summarize_research_interest(cfg: LlmConfig, text: str) -> Optional[str]:
	if not text:
		return None
	from .prompts import SUMMARIZE_RESEARCH_INTEREST_PROMPT
	
	client = _get_llm_client(cfg)
	if not client:
		return None

	try:
		res = client.chat.completions.create(
			model=cfg.model,
			temperature=0.1,
			max_tokens=1024,
			messages=[
				{"role": "system", "content": SUMMARIZE_RESEARCH_INTEREST_PROMPT},
				{"role": "user", "content": text},
			]
		)
		if res.choices:
			return res.choices[0].message.content.strip()
	except Exception as e:
		print(f"Error calling LLM for summary: {e}")
	return None


def refine_profile_info(cfg: LlmConfig, text: str) -> Optional[Dict[str, str]]:
	if not text:
		return None
	from .prompts import REFINE_PROFILE_PROMPT
	
	client = _get_llm_client(cfg)
	if not client:
		return None

	try:
		res = client.chat.completions.create(
			model=cfg.model,
			temperature=0.0,
			max_tokens=256,
			response_format={"type": "json_object"},
			messages=[
				{"role": "system", "content": REFINE_PROFILE_PROMPT},
				{"role": "user", "content": text},
			]
		)
		if res.choices:
			content = res.choices[0].message.content.strip()
			try:
				# The content should be a JSON string, parse it
				json_content = json.loads(content)
				return {
					"professor_name": json_content.get("professor_name"),
					"title": json_content.get("title"),
				}
			except json.JSONDecodeError:
				print(f"LLM returned invalid JSON for profile refinement: {content}")

	except Exception as e:
		print(f"Error calling LLM for profile refinement: {e}")
	return None
