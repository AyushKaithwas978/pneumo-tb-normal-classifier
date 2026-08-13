import re
from pathlib import Path


DEFAULT_DISCLAIMER = (
    "This assistant is for educational use only and is not a medical diagnosis. "
    "A qualified clinician should review symptoms, history, physical examination, "
    "and imaging before making medical decisions."
)


FOLLOW_UP_QUESTIONS = {
    "Normal": [
        "What symptoms should still be monitored if the X-ray appears normal?",
        "When should a patient seek medical attention despite a normal result?",
        "What other tests might a clinician consider if symptoms continue?",
    ],
    "Pneumonia": [
        "What symptoms commonly appear with pneumonia?",
        "What warning signs need urgent medical care?",
        "What follow-up tests might confirm or rule out pneumonia?",
    ],
    "Tuberculosis": [
        "What symptoms are commonly associated with tuberculosis?",
        "What tests are used to confirm tuberculosis?",
        "Why is medical follow-up important when TB is suspected?",
    ],
}


NEXT_STEPS = {
    "Normal": [
        "Use the prediction as a screening aid, not a diagnosis.",
        "If symptoms continue, consult a clinician for examination and additional testing.",
        "Seek urgent care for severe breathing difficulty, chest pain, confusion, or blue lips.",
    ],
    "Pneumonia": [
        "Consult a clinician, especially if fever, cough, chest pain, or breathing difficulty is present.",
        "A clinician may compare the image with symptoms, oxygen level, blood tests, or sputum tests.",
        "Seek urgent care for severe shortness of breath, persistent high fever, confusion, or low oxygen.",
    ],
    "Tuberculosis": [
        "Arrange medical follow-up promptly if TB is suspected.",
        "A clinician may request sputum testing, molecular testing, culture, or additional imaging.",
        "Avoid using this model result alone to decide treatment or isolation.",
    ],
}


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _split_markdown_sections(source, text):
    document_title = source
    current_title = source
    current_lines = []
    sections = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if heading:
            if current_lines:
                sections.append(
                    {
                        "source": source,
                        "title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            heading_text = heading.group(2).strip()
            if len(heading.group(1)) == 1:
                document_title = heading_text
                current_title = heading_text
            else:
                current_title = f"{document_title} - {heading_text}"
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "source": source,
                "title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


class KnowledgeBase:
    def __init__(self, knowledge_dir):
        self.knowledge_dir = Path(knowledge_dir)
        self.snippets = self._load_snippets()

    def _load_snippets(self):
        snippets = []
        if not self.knowledge_dir.exists():
            return snippets

        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            snippets.extend(_split_markdown_sections(path.name, text))
        return snippets

    def search(self, query, top_k=3):
        query_tokens = _tokenize(query)
        ranked = []

        for snippet in self.snippets:
            haystack = f"{snippet['source']} {snippet['title']} {snippet['text']}"
            snippet_tokens = _tokenize(haystack)
            overlap = len(query_tokens & snippet_tokens)
            exact_boost = 2 if query.lower() in haystack.lower() else 0
            overview_boost = 4 if snippet["title"].lower().endswith("overview") else 0
            score = overlap + exact_boost + overview_boost
            if score > 0:
                ranked.append((score, snippet))

        ranked.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["title"]))
        return [snippet for _, snippet in ranked[:top_k]]


def build_explanation(label, confidence, probabilities, knowledge_base):
    query = f"{label} chest x-ray symptoms findings next steps safety"
    snippets = knowledge_base.search(query, top_k=4)
    clinical_context = [snippet["text"] for snippet in snippets[:3]]
    citations = [
        {"source": snippet["source"], "title": snippet["title"]}
        for snippet in snippets
    ]

    confidence_pct = f"{confidence * 100:.2f}%"
    probability_lines = [
        f"{name}: {value * 100:.2f}%"
        for name, value in sorted(probabilities.items())
    ]

    return {
        "summary": (
            f"The model's top prediction is {label} with {confidence_pct} confidence."
        ),
        "probability_notes": probability_lines,
        "clinical_context": clinical_context
        or [
            "No matching local reference section was found for this label. "
            "Use the probabilities as a screening signal only."
        ],
        "next_steps": NEXT_STEPS.get(
            label,
            [
                "Review the image and symptoms with a qualified clinician.",
                "Do not use this result alone for medical decisions.",
                "Seek urgent care if severe or rapidly worsening symptoms are present.",
            ],
        ),
        "suggested_questions": FOLLOW_UP_QUESTIONS.get(
            label,
            [
                "What symptoms should be considered with this result?",
                "What additional tests might a clinician request?",
                "When should urgent medical care be considered?",
            ],
        ),
        "citations": citations,
        "disclaimer": DEFAULT_DISCLAIMER,
    }
