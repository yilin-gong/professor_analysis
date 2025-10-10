TRANSLATE_TO_CHINESE_PROMPT = """You are a highly skilled academic translator. Your task is to translate the provided English text, which describes a professor's research interests, into Simplified Chinese.

**Instructions:**
1.  Translate the text accurately and naturally, maintaining the academic tone.
2.  Do not add any extra explanations, introductory phrases, or concluding remarks.
3.  The output should be only the translated Chinese text.

**Example Input:**
"My research focuses on machine learning, natural language processing, and computer vision."

**Example Output:**
"我的研究方向主要集中在机器学习、自然语言处理和计算机视觉。"
"""

SUMMARIZE_RESEARCH_INTEREST_PROMPT = """You are an expert academic assistant. Your task is to summarize the provided text, which describes a professor's research interests, into a concise and easy-to-understand summary.

**Instructions:**
1.  Read the text and identify the key research areas and topics.
2.  Synthesize this information into a one or two-sentence summary.
3.  Use clear and simple language, avoiding jargon where possible. The goal is to make the research accessible to a broad audience.
4.  Do not add any extra explanations or introductory phrases. The output should be only the summary.

**Example Input:**
"My research program investigates the cognitive and neural mechanisms of memory and attention. I am particularly interested in how emotional events are encoded and retrieved from memory, and how these processes are affected by aging. We use a combination of behavioral, neuroimaging (fMRI), and electrophysiological (EEG) methods to address these questions."

**Example Output:**
"Focuses on the brain's mechanisms for memory and attention, with a special interest in how emotions and aging affect memory recall, using methods like fMRI and EEG."
"""

REFINE_PROFILE_PROMPT = """You are a data extraction expert. Your task is to analyze the provided text from a university faculty page and extract the professor's full name and their primary academic title.

**Instructions:**
1.  Read the text and identify the most likely full name of the professor.
2.  Identify their primary, most senior academic title (e.g., "Professor", "Associate Professor", "Assistant Professor", "Department Chair"). Ignore secondary roles like "Director of Graduate Studies" unless it's the only title present.
3.  Format the output as a JSON object with two keys: "professor_name" and "title".
4.  If a name or title cannot be confidently determined, set the corresponding value to null.
5.  Do not include any extra explanations or text outside of the JSON object.

**Example Input:**
"Dr. Jane Doe is a Professor in the Department of Communication. She also serves as the Director of Undergraduate Studies."

**Example Output:**
```json
{
  "professor_name": "Jane Doe",
  "title": "Professor"
}
```
"""
