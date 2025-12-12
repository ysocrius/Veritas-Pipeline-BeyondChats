import re
from typing import List

# A lightweight list of toxic patterns for demonstration.
# In production, this would be replaced or augmented by a model like 'unitary/toxic-bert'
# but for high-throughput/low-cost "Guardrails", regex is often the first line of defense.
TOXIC_PATTERNS = [
    r"hate\s?speech",
    r"kill\s?yourself",
    r"violent",
    r"racist",
    r"idiot",
    r"stupid",
    # Add more patterns as needed
]

class ToxicityGuardrail:
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def check(self, text: str) -> float:
        """
        Returns a score from 0.0 (Safe) to 1.0 (Toxic).
        """
        for pattern in self.patterns:
            if pattern.search(text):
                return 1.0 # Flagged as toxic
        return 0.0 # Safe

_guardrail = None

def score_toxicity(text: str) -> float:
    """
    Evaluates the text for potential toxicity/safety violations.
    High score = High Toxicity (Bad).
    """
    global _guardrail
    if _guardrail is None:
        _guardrail = ToxicityGuardrail()
    
    return _guardrail.check(text)
