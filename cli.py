import argparse
import csv
import json
import os
import time
import re
from typing import List, Optional
from urllib.parse import urlparse, urlunparse, ParseResult
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from .http_client import fetch_html_auto
from .regex_filters import extract_link_candidates
from .models import LlmConfig, ProfileResult, LinkCandidate
from .parse_profile import heuristic_analyze, refine_with_llm
from .list_page_parser import extract_profiles_from_listing
from .llm import translate_to_chinese
from .llm import summarize_research_interest


NAME_LIKE_PAT = re.compile(r"\b([A-Z][a-z]{1,})(\s+[A-Z][a-z]{1,}){1,3}\b")
PROFILE_URL_PATH_PAT = re.compile(r"/(?:staff|faculty|people|directory|profile|faculty-research)/[^/]+/?$", re.I)
LISTING_PAGE_PAT = re.compile(r"/people/[^/]*directory\.html$", re.I)


def _process_candidate(cand: LinkCandidate, university: str, program: str, use_browser: bool, source_url: str, cfg: LlmConfig) -> Optional[ProfileResult]:
	"""Fetches and analyzes a single candidate profile."""
	html, final_url = fetch_html_auto(cand.url, use_browser=use_browser)
	if not html or not final_url:
		return None
	
	res = heuristic_analyze(final_url, html, university, program, llm_config=cfg)
	if res and not res.professor_name:
		res.professor_name = cand.text
	return res


def _read_urls(args: argparse.Namespace) -> List[str]:
	urls: List[str] = []
	if args.urls:
		for u in args.urls:
			urls.extend([s for s in u.split(',') if s.strip()])
	if args.from_file:
		with open(args.from_file, 'r') as f:
			for line in f:
				line = line.strip()
				if line and not line.startswith('#'):
					urls.append(line)
	if not urls:
		env_urls = os.getenv('URLS')
		if env_urls:
			urls = [s.strip() for s in env_urls.split(',') if s.strip()]
	return [_normalize_directory_url(u.strip()) for u in urls if u.strip()]


def _normalize_directory_url(u: str) -> str:
	pr = urlparse(u)
	frag = pr.fragment or ""
	page_m = re.search(r"(?:^|[&#])page=(\d+)", frag)
	if page_m:
		page = page_m.group(1)
		new_query = f"page={page}"
		clean = ParseResult(pr.scheme, pr.netloc, pr.path, pr.params, new_query, "")
		return urlunparse(clean)
	clean = ParseResult(pr.scheme, pr.netloc, pr.path, pr.params, pr.query, "")
	return urlunparse(clean)


def _write_csv(path: str, results: List[ProfileResult]) -> None:
	fieldnames = [
		"university_name", "program_name", "professor_name", "title", "profile_url", "research_interest", "research_interest_summary", "research_interest_chinese", "keywords", "contact_info", "is_professor_page", "is_accepting_phd"
	]
	with open(path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in results:
			writer.writerow({
				"university_name": r.university_name or "",
				"program_name": r.program_name or "",
				"professor_name": r.professor_name or "",
				"title": r.title or "",
				"profile_url": r.profile_url,
				"research_interest": r.research_interest or "none",
				"research_interest_summary": r.research_interest_summary or "",
				"research_interest_chinese": r.research_interest_chinese or "",
				"keywords": "; ".join(r.keywords),
				"contact_info": "; ".join(r.contact_info.emails),
				"is_professor_page": r.is_professor_page,
				"is_accepting_phd": "unknown" if r.is_accepting_phd is None else bool(r.is_accepting_phd),
			})


def _write_json(path: str, results: List[ProfileResult]) -> None:
	with open(path, 'w') as f:
		json.dump([r.model_dump() for r in results], f, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
	load_dotenv()
	parser = argparse.ArgumentParser(description="Professor directory scraper")
	parser.add_argument('--urls', nargs='*', help='One or more directory URLs (or set URLS env)')
	parser.add_argument('--from-file', help='Path to file containing URLs, one per line')
	parser.add_argument('--university', help='University name (optional; env UNIVERSITY or auto-detect)')
	parser.add_argument('--program', help='Program name (optional; env PROGRAM or auto-detect)')
	parser.add_argument('--no-llm', action='store_true', help='Disable LLM usage (or set NO_LLM=1)')
	parser.add_argument('--max-profiles', type=int, help='Max profile pages to process (or MAX_PROFILES env)')
	parser.add_argument('--out-csv', help='CSV output path (or OUT_CSV env)')
	parser.add_argument('--out-json', help='JSON output path (optional; OUT_JSON env)')
	parser.add_argument('--model', help='Override LLM model ID (e.g., doubao-...)')
	parser.add_argument('--base-url', help='Override LLM base URL')
	parser.add_argument('--use-browser', action='store_true', help='Use headless browser (Playwright) to bypass challenges (or USE_BROWSER=1)')
	args = parser.parse_args(argv)

	urls = _read_urls(args)
	if not urls:
		print('No URLs provided (set --urls or URLS env)')
		return 2

	university = args.university or os.getenv('UNIVERSITY')
	program = args.program or os.getenv('PROGRAM')
	out_csv = args.out_csv or os.getenv('OUT_CSV', 'results.csv')
	out_json = args.out_json or os.getenv('OUT_JSON')
	max_profiles = args.max_profiles or int(os.getenv('MAX_PROFILES', '100'))
	no_llm_flag = args.no_llm or os.getenv('NO_LLM', '').lower() in ('1','true','yes')
	use_browser = args.use_browser or os.getenv('USE_BROWSER', '').lower() in ('1','true','yes') or True

	cfg = LlmConfig(
		enabled=(not no_llm_flag),
		api_key=os.getenv('OPENAI_API_KEY') or os.getenv('ARK_API_KEY'),
		base_url=(args.base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('ARK_BASE_URL')),
		model=(args.model or os.getenv('LLM_MODEL') or 'gpt-4o-mini'),
	)

	all_results: List[ProfileResult] = []
	seen_profiles = set()
	for directory_url in urls:
		print(f"Fetching directory: {directory_url}")
		html, _ = fetch_html_auto(directory_url, use_browser=use_browser)
		if not html:
			print(f"Failed to fetch directory page: {directory_url}")
			continue

		# Skip individual profile check for known directory pages
		is_known_directory = any(pattern in directory_url for pattern in [
			"our-people", "faculty-staff", "directory", "people/faculty", "/faculty/"
		])
		
		if not is_known_directory:
			# If the page itself is a profile, analyze it directly
			res_probe = heuristic_analyze(directory_url, html, university, program, llm_config=cfg)
			if res_probe.is_professor_page:
				print("Page appears to be an individual profile. Analyzing directly.")
				if cfg.enabled and (not res_probe.research_interest or not res_probe.university_name or not res_probe.program_name):
					res_probe = refine_with_llm(cfg, directory_url, html, res_probe)
				all_results.append(res_probe)
				print(f"  -> Heuristic: is_prof={res_probe.is_professor_page}, name='{res_probe.professor_name or '-'}', research={bool(res_probe.research_interest)}")
				continue

		# Otherwise treat as listing/directory
		# Directory context
		dir_university = university
		dir_program = program
		if not dir_university or not dir_program:
			from bs4 import BeautifulSoup
			soup_dir = BeautifulSoup(html, 'lxml')
			title_dir = (soup_dir.find('title').get_text(' ') if soup_dir.find('title') else '').strip()
			crumb = soup_dir.select_one('.breadcrumb')
			if not dir_university:
				dir_university = title_dir or dir_university
			if not dir_program and crumb:
				dir_program = crumb.get_text(' ').strip()

		candidates = extract_link_candidates(html, directory_url)
		
		# Filter out already seen profiles and limit to max_profiles
		new_candidates = []
		for c in candidates:
			if c.url not in seen_profiles:
				new_candidates.append(c)
		
		total_candidates_to_process = min(max_profiles - len(all_results), len(new_candidates))
		candidates_to_process = new_candidates[:total_candidates_to_process]
		
		print(f"Found {len(candidates)} total candidates ({len(new_candidates)} new). Processing {len(candidates_to_process)}.")

		processed_on_page = 0
		has_profile_links = any(PROFILE_URL_PATH_PAT.search(urlparse(c.url).path or "") for c in candidates)
		if not has_profile_links:
			listing_profiles = extract_profiles_from_listing(html, directory_url, dir_university, dir_program)
			print(f"Listing extraction fallback: {len(listing_profiles)} entries found.")
			for res in listing_profiles:
				if len(all_results) >= max_profiles:
					break
				# If the listing provides a real profile link, fetch and analyze it
				is_same_page_anchor = res.profile_url.startswith(_normalize_directory_url(directory_url) + "#")
				if not is_same_page_anchor:
					page_html, final_url = fetch_html_auto(res.profile_url, use_browser=use_browser, referer=directory_url)
					if page_html:
						analyzed = heuristic_analyze(final_url or res.profile_url, page_html, dir_university, dir_program, llm_config=cfg)
						if cfg.enabled and (not analyzed.research_interest or not analyzed.university_name or not analyzed.program_name):
							analyzed = refine_with_llm(cfg, final_url or res.profile_url, page_html, analyzed)
						res = analyzed
				if res.profile_url in seen_profiles:
					continue
				seen_profiles.add(res.profile_url)
				print(f"From listing: {res.professor_name} ({res.title or '-'}) -> {res.contact_info.emails[:1]}")
				all_results.append(res)
				processed_on_page += 1
			print(f"Analyzed {processed_on_page} entries from listing.")
			continue

		# Process candidates in parallel
		with ThreadPoolExecutor(max_workers=8) as executor:
			future_to_cand = {executor.submit(_process_candidate, cand, dir_university, dir_program, use_browser, directory_url, cfg): cand for cand in candidates_to_process}
			
			for future in tqdm(as_completed(future_to_cand), total=len(candidates_to_process), desc="Analyzing profiles"):
				cand = future_to_cand[future]
				seen_profiles.add(cand.url)
				try:
					res = future.result()
					if res:
						all_results.append(res)
						processed_on_page += 1
				except Exception as exc:
					print(f'{cand.url} generated an exception: {exc}')

		print(f"Analyzed {processed_on_page} new profiles from this page.")

	if all_results:
		print("Translating research interests to Chinese...")
		results_to_translate = [res for res in all_results if res.research_interest]
		with ThreadPoolExecutor(max_workers=8) as executor:
			future_to_res = {executor.submit(translate_to_chinese, cfg, res.research_interest): res for res in results_to_translate}
			for future in tqdm(as_completed(future_to_res), total=len(results_to_translate), desc="Translating"):
				res_obj = future_to_res[future]
				try:
					translation = future.result()
					if translation:
						res_obj.research_interest_chinese = translation
				except Exception as exc:
					print(f'Translation for "{res_obj.professor_name}" generated an exception: {exc}')

	if all_results:
		print("Summarizing research interests...")
		results_to_summarize = [res for res in all_results if res.research_interest]
		with ThreadPoolExecutor(max_workers=8) as executor:
			future_to_res = {executor.submit(summarize_research_interest, cfg, res.research_interest): res for res in results_to_summarize}
			for future in tqdm(as_completed(future_to_res), total=len(results_to_summarize), desc="Summarizing"):
				res_obj = future_to_res[future]
				try:
					summary = future.result()
					if summary:
						res_obj.research_interest_summary = summary
				except Exception as exc:
					print(f'Summary for "{res_obj.professor_name}" generated an exception: {exc}')


	print(f"\n--- CRAWL COMPLETE ---")
	print(f"Total unique profiles analyzed: {len(all_results)}")
	print(f"Writing CSV to {out_csv} ({len(all_results)} rows)")
	_write_csv(out_csv, all_results)
	if out_json:
		_write_json(out_json, all_results)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
