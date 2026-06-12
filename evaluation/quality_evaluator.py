"""Lightweight deterministic quality evaluation.

The evaluator uses FP32 model outputs as local references and scores compressed
model outputs with string similarity, length matching, and simple completeness
heuristics. It does not call external APIs or use an LLM judge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS_PATH = Path("evaluation/prompts.json")
DEFAULT_MAX_NEW_TOKENS = 96


class QualityEvaluator:
    """Runs prompt-based quality checks against FP32 reference outputs."""

    def __init__(
        self,
        prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.prompts_path = Path(prompts_path)
        self.max_new_tokens = max_new_tokens
        self.prompts = self.load_prompts(self.prompts_path)

    @staticmethod
    def load_prompts(prompts_path: str | Path) -> list[dict[str, str]]:
        """Load evaluation prompts from JSON."""

        raw_prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
        prompts: list[dict[str, str]] = []

        for index, item in enumerate(raw_prompts, start=1):
            if isinstance(item, str):
                prompt = {"id": f"quality_{index:03d}", "text": item}
            else:
                prompt = {
                    "id": str(item.get("id", f"quality_{index:03d}")),
                    "text": str(item["text"]),
                }
            if not prompt["text"].strip():
                raise ValueError(f"Quality prompt {prompt['id']} is empty.")
            prompts.append(prompt)

        if not prompts:
            raise ValueError("Quality evaluation requires at least one prompt.")
        return prompts

    def build_references(self, model: Any, tokenizer: Any, device: str) -> dict[str, str]:
        """Generate FP32 reference responses for every quality prompt."""

        return {
            prompt["id"]: self.generate_response(model, tokenizer, prompt["text"], device)
            for prompt in self.prompts
        }

    def evaluate_baseline(self, references: dict[str, str]) -> dict[str, Any]:
        """Return a perfect self-reference score for the FP32 baseline."""

        prompt_scores = [
            {
                "id": prompt["id"],
                "similarity_score": 1.0,
                "length_score": 1.0,
                "completeness_score": self.completeness_score(references[prompt["id"]]),
                "quality_score": 1.0,
            }
            for prompt in self.prompts
        ]
        return {
            "quality_score": 1.0,
            "quality_prompt_count": len(prompt_scores),
            "quality_prompt_scores": prompt_scores,
        }

    def evaluate_model(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        references: dict[str, str],
    ) -> dict[str, Any]:
        """Generate model outputs and score them against FP32 references."""

        prompt_scores: list[dict[str, Any]] = []

        for prompt in self.prompts:
            response = self.generate_response(model, tokenizer, prompt["text"], device)
            reference = references[prompt["id"]]
            score = self.score_response(response, reference)
            prompt_scores.append(
                {
                    "id": prompt["id"],
                    "similarity_score": score["similarity_score"],
                    "length_score": score["length_score"],
                    "completeness_score": score["completeness_score"],
                    "quality_score": score["quality_score"],
                }
            )

        quality_score = sum(item["quality_score"] for item in prompt_scores) / len(prompt_scores)
        return {
            "quality_score": round(quality_score, 4),
            "quality_prompt_count": len(prompt_scores),
            "quality_prompt_scores": prompt_scores,
        }

    def generate_response(self, model: Any, tokenizer: Any, prompt: str, device: str) -> str:
        """Generate a deterministic answer for one quality prompt."""

        from benchmark.comparison import load_torch

        torch = load_torch()
        generation_model = self._unwrap_model(model)
        generation_device = self._model_device(generation_model, device)
        inputs = tokenizer(prompt, return_tensors="pt").to(generation_device)

        with torch.inference_mode():
            output_ids = generation_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()

        input_token_count = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_token_count:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def score_response(self, response: str, reference: str) -> dict[str, float]:
        """Score one response against the FP32 reference."""

        similarity = self.similarity_score(response, reference)
        length = self.length_score(response, reference)
        completeness = self.completeness_score(response)
        quality = (0.60 * similarity) + (0.25 * length) + (0.15 * completeness)

        return {
            "similarity_score": round(similarity, 4),
            "length_score": round(length, 4),
            "completeness_score": round(completeness, 4),
            "quality_score": round(quality, 4),
        }

    @staticmethod
    def similarity_score(response: str, reference: str) -> float:
        """Blend token overlap and character-level similarity."""

        response_tokens = _tokenize(response)
        reference_tokens = _tokenize(reference)
        if not response_tokens or not reference_tokens:
            return 0.0

        overlap = Counter(response_tokens) & Counter(reference_tokens)
        overlap_count = sum(overlap.values())
        precision = overlap_count / len(response_tokens)
        recall = overlap_count / len(reference_tokens)
        token_f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        sequence_ratio = SequenceMatcher(None, response.lower(), reference.lower()).ratio()
        return (0.70 * token_f1) + (0.30 * sequence_ratio)

    @staticmethod
    def length_score(response: str, reference: str) -> float:
        """Reward responses with length similar to the FP32 reference."""

        response_length = len(_tokenize(response))
        reference_length = len(_tokenize(reference))
        if response_length == 0 or reference_length == 0:
            return 0.0
        return min(response_length, reference_length) / max(response_length, reference_length)

    @staticmethod
    def completeness_score(response: str) -> float:
        """Estimate whether the response is non-empty and not obviously cut off."""

        tokens = _tokenize(response)
        if not tokens:
            return 0.0

        score = 0.45
        if len(tokens) >= 12:
            score += 0.35
        elif len(tokens) >= 5:
            score += 0.20

        stripped = response.strip()
        if stripped.endswith((".", "!", "?", ":", ";", ")")):
            score += 0.20
        elif len(tokens) >= 20:
            score += 0.10

        return min(score, 1.0)

    @staticmethod
    def _model_device(model: Any, fallback_device: str) -> str:
        try:
            return str(next(model.parameters()).device)
        except StopIteration:
            return fallback_device

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        if hasattr(model, "generate"):
            return model
        return getattr(model, "model", model)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())
