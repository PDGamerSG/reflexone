"""Validated input and output contracts; schema validity is not semantic correctness."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexone.data import load_ontology

ONTOLOGY = load_ontology()
IntentLabel = Enum("IntentLabel", {name: name for name in ONTOLOGY.intents}, type=str)
SlotLabel = Enum("SlotLabel", {name: name for name in ONTOLOGY.slot_types}, type=str)
Domain = Literal["banking", "hotels"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["support", "commands", "spam"] = "support"
    text: str = Field(min_length=1, max_length=4000)
    domain: Domain | None = None

    @model_validator(mode="after")
    def validate_request(self) -> PredictRequest:
        if not self.text.strip():
            raise ValueError("text cannot be blank")
        if self.profile == "support" and self.domain is None:
            raise ValueError("support requires domain='banking' or domain='hotels'")
        if self.profile != "support" and self.domain is not None:
            raise ValueError("domain is only applicable to the support profile")
        return self


class IntentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: bool
    probability: Probability


class ExtractedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: SlotLabel
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    token_score: Probability

    @model_validator(mode="after")
    def valid_bounds(self) -> ExtractedSpan:
        if self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("Span must have valid half-open bounds matching its text length")
        return self


class SupportDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["support"] = "support"
    model_id: str
    domain: Domain
    text: str
    intents: dict[IntentLabel, IntentPrediction]
    slots: list[ExtractedSpan]
    review_required: bool
    review_reasons: list[Literal["low_intent_margin", "no_intent_selected", "input_truncated"]]
    truncated: bool

    @model_validator(mode="after")
    def validate_decision(self) -> SupportDecision:
        expected = {
            name
            for name, allowed in zip(
                ONTOLOGY.intents, ONTOLOGY.intent_mask(self.domain), strict=True
            )
            if allowed
        }
        if set(self.intents) != expected:
            raise ValueError("Intent keys must be exactly the domain's known ontology labels")
        allowed_slots = {
            name
            for name, allowed in zip(
                ONTOLOGY.slot_types, ONTOLOGY.slot_mask(self.domain), strict=True
            )
            if allowed
        }
        for span in self.slots:
            if span.label not in allowed_slots:
                raise ValueError("Slot type is not allowed for this domain")
            if self.text[span.start : span.end] != span.text:
                raise ValueError("Extracted text must exactly match the original input slice")
        if self.review_required != bool(self.review_reasons):
            raise ValueError("review_required must agree with review_reasons")
        if self.truncated and "input_truncated" not in self.review_reasons:
            raise ValueError("Truncated requests require review")
        return self


class CategoricalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["commands", "spam"]
    label: str
    candidate_label: str
    probabilities: dict[str, Probability]
    confidence: Probability
    abstain: bool
    reason: str | None
    model: str

    @model_validator(mode="after")
    def validate_class(self) -> CategoricalDecision:
        if self.candidate_label not in self.probabilities:
            raise ValueError("candidate_label must be one of the known classes")
        if abs(sum(self.probabilities.values()) - 1) > 1e-5:
            raise ValueError("Categorical probabilities must sum to one")
        if self.label != self.candidate_label and not (
            self.task == "commands" and self.label == "oos" and self.abstain
        ):
            raise ValueError("label must be a known class or an abstained command")
        if self.task == "spam" and set(self.probabilities) != {"ham", "spam"}:
            raise ValueError("Spam classes must be ham and spam")
        return self
