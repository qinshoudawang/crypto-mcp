from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .event_types import EventType
from .models import ContentItem, Entities
from .taxonomy_rules import (
    CHAIN_KEYWORDS,
    DEFAULT_DYNAMIC_ALIAS_PATH,
    ENTITY_ALIASES,
    EVENT_TYPE_RULES,
    INSTITUTIONAL_KEYWORDS,
    INSTITUTION_NAMES,
    LAUNCH_KEYWORDS,
    MACRO_KEYWORDS,
    MARKET_STRUCTURE_KEYWORDS,
    NON_CRYPTO_NOISE_KEYWORDS,
    TITLE_EVENT_HINTS,
    TITLE_MARKET_STRUCTURE_HINTS,
    TOKEN_ALIASES,
    TOPIC_ALIASES,
    TOPIC_EVENT_TYPE_MAP,
    TOPIC_KEYWORDS,
)


class ContentNormalizer:
    DEFAULT_DYNAMIC_ALIAS_PATH = DEFAULT_DYNAMIC_ALIAS_PATH
    CHAIN_KEYWORDS = CHAIN_KEYWORDS
    TOPIC_KEYWORDS = TOPIC_KEYWORDS
    TITLE_EVENT_HINTS = TITLE_EVENT_HINTS
    TOKEN_ALIASES = TOKEN_ALIASES
    ENTITY_ALIASES = ENTITY_ALIASES
    TOPIC_ALIASES = TOPIC_ALIASES
    MACRO_KEYWORDS = MACRO_KEYWORDS
    NON_CRYPTO_NOISE_KEYWORDS = NON_CRYPTO_NOISE_KEYWORDS

    def __init__(self, dynamic_alias_path: str | None = None) -> None:
        self.chain_keywords = list(self.CHAIN_KEYWORDS)
        self.topic_keywords = list(self.TOPIC_KEYWORDS)
        self.title_event_hints = dict(self.TITLE_EVENT_HINTS)
        self.token_aliases = {key: list(value) for key, value in self.TOKEN_ALIASES.items()}
        self.entity_aliases = {key: list(value) for key, value in self.ENTITY_ALIASES.items()}
        self.topic_aliases = {key: list(value) for key, value in self.TOPIC_ALIASES.items()}
        self.macro_keywords = list(self.MACRO_KEYWORDS)
        self.non_crypto_noise_keywords = list(self.NON_CRYPTO_NOISE_KEYWORDS)
        self.dynamic_alias_path = dynamic_alias_path or os.getenv(
            "FOLLOWIN_DYNAMIC_ALIAS_PATH",
            self.DEFAULT_DYNAMIC_ALIAS_PATH,
        )
        self._load_dynamic_aliases()

    def normalize(self, raw: Dict[str, Any]) -> ContentItem:
        title = raw.get("translated_title") or raw.get("title") or ""
        content = (
            raw.get("translated_content")
            or raw.get("content")
            or raw.get("translated_full_content")
            or raw.get("full_content")
            or ""
        )
        published_at = self._parse_datetime(raw.get("publish_time"))
        item_id = str(raw.get("id") or self._stable_id(title + (raw.get("source_url") or "")))

        item = ContentItem(
            id=item_id,
            title=title,
            summary=self._make_summary(content),
            content=content,
            url=raw.get("source_url") or raw.get("page_url") or raw.get("jump_url", ""),
            source_type=raw.get("source_name", "unknown"),
            source_name=raw.get("source_title") or raw.get("source_name", "unknown"),
            author=raw.get("nickname") or raw.get("username") or "unknown",
            published_at=published_at,
            language=raw.get("source_lang", raw.get("language", "unknown")),
            raw_tags=[
                tag.get("symbol", "") or tag.get("name", "")
                for tag in raw.get("tags", [])
                if isinstance(tag, dict)
            ],
        )

        item.entities = self.extract_entities(item, raw)
        item.event_type = self.classify_event_type(item)
        item.credibility_score = self.assess_credibility(item)
        item.importance_score = self.assess_importance(item)
        return item

    def extract_entities(self, item: ContentItem, raw: Dict[str, Any]) -> Entities:
        entities = Entities()
        entity_sources: Dict[str, Dict[str, List[str]]] = {
            "projects": {},
            "tokens": {},
            "chains": {},
            "topics": {},
            "people": {},
        }

        tags = raw.get("tags", [])
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            tag_type = (tag.get("type") or "").lower()
            name = (tag.get("name") or "").strip()
            symbol = (tag.get("symbol") or "").strip()
            if tag_type == "token":
                if name:
                    entities.projects.append(name)
                    self._record_entity_source(entity_sources, "projects", name, "tag:name")
                if symbol:
                    entities.tokens.append(symbol.upper())
                    self._record_entity_source(entity_sources, "tokens", symbol.upper(), "tag:symbol")

        title_text = item.title.lower()
        body_text = item.content.lower()
        full_text = f"{title_text} {body_text}".strip()

        for chain in self.chain_keywords:
            title_hit = self._contains_alias(title_text, chain)
            body_hit = self._contains_alias(body_text, chain)
            if title_hit or body_hit:
                normalized_chain = chain.title()
                entities.chains.append(normalized_chain)
                if title_hit:
                    self._record_entity_source(
                        entity_sources, "chains", normalized_chain, f"title:{chain}"
                    )
                if body_hit:
                    self._record_entity_source(
                        entity_sources, "chains", normalized_chain, f"body:{chain}"
                    )

        for topic in self.topic_keywords:
            title_hit = self._contains_topic_alias(title_text, topic)
            body_hit = self._contains_topic_alias(body_text, topic)
            if title_hit or body_hit:
                normalized_topic = self._title_case_topic(topic)
                entities.topics.append(normalized_topic)
                if title_hit:
                    self._record_entity_source(
                        entity_sources, "topics", normalized_topic, f"title:{topic}"
                    )
                if body_hit:
                    self._record_entity_source(
                        entity_sources, "topics", normalized_topic, f"body:{topic}"
                    )

        for canonical_name, aliases in self.entity_aliases.items():
            matched_aliases = [
                alias
                for alias in aliases
                if self._contains_alias(title_text, alias) or self._contains_alias(body_text, alias)
            ]
            if matched_aliases:
                entities.projects.append(canonical_name)
                for alias in matched_aliases:
                    if self._contains_alias(title_text, alias):
                        self._record_entity_source(
                            entity_sources, "projects", canonical_name, f"title:{alias}"
                        )
                    if self._contains_alias(body_text, alias):
                        self._record_entity_source(
                            entity_sources, "projects", canonical_name, f"body:{alias}"
                        )

        for symbol, aliases in self.token_aliases.items():
            matched_aliases = [
                alias
                for alias in aliases
                if self._contains_alias(title_text, alias) or self._contains_alias(body_text, alias)
            ]
            if matched_aliases:
                entities.tokens.append(symbol)
                for alias in matched_aliases:
                    if self._contains_alias(title_text, alias):
                        self._record_entity_source(
                            entity_sources, "tokens", symbol, f"title:{alias}"
                        )
                    if self._contains_alias(body_text, alias):
                        self._record_entity_source(
                            entity_sources, "tokens", symbol, f"body:{alias}"
                        )

        for canonical_topic, aliases in self.topic_aliases.items():
            matched_aliases = [
                alias
                for alias in aliases
                if self._contains_topic_alias(full_text, alias)
            ]
            if matched_aliases:
                entities.topics.append(canonical_topic)
                for alias in matched_aliases:
                    if self._contains_topic_alias(title_text, alias):
                        self._record_entity_source(
                            entity_sources, "topics", canonical_topic, f"title:{alias}"
                        )
                    if self._contains_topic_alias(body_text, alias):
                        self._record_entity_source(
                            entity_sources, "topics", canonical_topic, f"body:{alias}"
                        )

        entities.projects = self._dedupe_preserve_order(entities.projects)
        entities.tokens = self._dedupe_preserve_order(entities.tokens)
        entities.chains = self._dedupe_preserve_order(entities.chains)
        entities.topics = self._dedupe_preserve_order(self._normalize_topics(entities.topics))
        entities.people = list(dict.fromkeys(entities.people))
        item.entity_sources = self._normalize_entity_sources(entity_sources)
        item.entity_confidence = self._build_entity_confidence(item.entity_sources)
        return entities

    def classify_event_type(self, item: ContentItem) -> EventType:
        text = f"{item.title} {item.content}".lower()
        title_text = item.title.lower()
        is_crypto = self._has_crypto_entity_signal(item)
        has_macro_signal = any(self._contains_alias(text, keyword) for keyword in self.macro_keywords)
        has_noise_signal = any(keyword in text for keyword in self.non_crypto_noise_keywords)

        if has_macro_signal and not is_crypto:
            return EventType.MACRO

        if has_noise_signal and not is_crypto:
            return EventType.UNKNOWN

        scores = self._build_event_type_scores(item, text, title_text, is_crypto, has_macro_signal, has_noise_signal)
        best_event_type, best_score = max(scores.items(), key=lambda item_score: item_score[1])
        if best_score >= 0.35:
            return best_event_type
        if has_macro_signal:
            return EventType.MACRO
        return EventType.UNKNOWN

    def assess_credibility(self, item: ContentItem) -> float:
        source_name = (item.source_type or "").lower()
        if source_name in {"twitter", "x"}:
            return 0.60
        if source_name in {"media"}:
            return 0.80
        if source_name in {"official"}:
            return 0.95
        return 0.50

    def assess_importance(self, item: ContentItem) -> float:
        score = 0.20
        score += self.title_event_hints.get(item.event_type, 0.0)

        entity_confidence = item.entity_confidence
        strong_projects = self._count_confidence_level(entity_confidence.get("projects", {}), "strong")
        strong_tokens = self._count_confidence_level(entity_confidence.get("tokens", {}), "strong")
        strong_topics = self._count_confidence_level(entity_confidence.get("topics", {}), "strong")

        if strong_projects:
            score += 0.18
        elif item.entities.projects:
            score += 0.10

        if strong_tokens:
            score += 0.10
        elif item.entities.tokens:
            score += 0.05

        if strong_topics:
            score += 0.08
        elif item.entities.topics:
            score += 0.04

        if self._has_title_entity_hit(item.entity_sources):
            score += 0.08

        score += (item.credibility_score - 0.5) * 0.24
        return round(self._clamp(score), 2)

    def _make_summary(self, text: str, max_len: int = 120) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned[:max_len] + ("..." if len(cleaned) > max_len else "")

    def _parse_datetime(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            if value > 10_000_000_000:
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _stable_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

    def _contains_alias(self, text: str, alias: str) -> bool:
        normalized_alias = alias.lower().strip()
        if not normalized_alias:
            return False
        if re.search(r"[a-z0-9]", normalized_alias):
            pattern = rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])"
            return re.search(pattern, text) is not None
        return normalized_alias in text

    def _contains_topic_alias(self, text: str, alias: str) -> bool:
        normalized_alias = alias.lower().strip()
        if normalized_alias == "agent":
            return self._contains_ai_agent_context(text)
        if normalized_alias == "yield":
            return self._contains_defi_yield_context(text)
        return self._contains_alias(text, normalized_alias)

    def _contains_ai_agent_context(self, text: str) -> bool:
        strong_patterns = [
            "ai agents",
            "ai agent",
            "agent skills",
            "agentic",
            "autonomous agent",
            "onchain agent",
            "on-chain agent",
        ]
        if any(self._contains_alias(text, pattern) for pattern in strong_patterns):
            return True

        guarded_patterns = [
            r"\bagent\b.{0,20}\bai\b",
            r"\bai\b.{0,20}\bagent\b",
            r"\bagent\b.{0,20}\bwallet\b",
            r"\bagent\b.{0,20}\bon[- ]?chain\b",
        ]
        return any(re.search(pattern, text) is not None for pattern in guarded_patterns)

    def _contains_defi_yield_context(self, text: str) -> bool:
        blocked_patterns = [
            "treasury yield",
            "bond yield",
            "yield curve",
            "10-year yield",
            "2-year yield",
        ]
        if any(self._contains_alias(text, pattern) for pattern in blocked_patterns):
            return False

        defi_patterns = [
            "yield farming",
            "yield farm",
            "yield-bearing",
            "yield vault",
            "defi yield",
            "staking yield",
            "lending yield",
        ]
        if any(self._contains_alias(text, pattern) for pattern in defi_patterns):
            return True

        guarded_patterns = [
            r"\byield\b.{0,20}\bfarm",
            r"\byield\b.{0,20}\bdefi\b",
            r"\byield\b.{0,20}\blending\b",
            r"\byield\b.{0,20}\bstaking\b",
        ]
        return any(re.search(pattern, text) is not None for pattern in guarded_patterns)

    def _normalize_topics(self, topics: List[str]) -> List[str]:
        normalized: List[str] = []
        for topic in topics:
            key = topic.lower().strip()
            if key == "defi":
                normalized.append("DeFi")
            elif key == "ai agents":
                normalized.append("AI Agents")
            elif key == "nft":
                normalized.append("NFT")
            elif key == "governance":
                normalized.append("Governance")
            elif key == "airdrop":
                normalized.append("Airdrop")
            elif key == "exploit":
                normalized.append("Exploit")
            elif key == "listing":
                normalized.append("Listing")
            elif key == "funding":
                normalized.append("Funding")
            elif key == "memecoin":
                normalized.append("Memecoin")
            elif key == "token unlock":
                normalized.append("Token Unlock")
            elif key == "partnership":
                normalized.append("Partnership")
            elif key == "merger":
                normalized.append("Merger")
            elif key == "incentive program":
                normalized.append("Incentive Program")
            elif key == "restaking":
                normalized.append("Restaking")
            elif key == "stablecoin":
                normalized.append("Stablecoin")
            elif key == "rwa":
                normalized.append("RWA")
            elif key == "derivatives":
                normalized.append("Derivatives")
            elif key == "layer2":
                normalized.append("Layer2")
            else:
                normalized.append(self._title_case_topic(topic))
        return normalized

    def _title_case_topic(self, topic: str) -> str:
        if topic.isupper():
            return topic
        return topic.title()

    def _dedupe_preserve_order(self, values: List[str]) -> List[str]:
        return list(dict.fromkeys(values))

    def _record_entity_source(
        self,
        entity_sources: Dict[str, Dict[str, List[str]]],
        bucket: str,
        entity: str,
        source: str,
    ) -> None:
        entity_sources.setdefault(bucket, {}).setdefault(entity, [])
        if source not in entity_sources[bucket][entity]:
            entity_sources[bucket][entity].append(source)

    def _normalize_entity_sources(
        self,
        entity_sources: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Dict[str, List[str]]]:
        normalized: Dict[str, Dict[str, List[str]]] = {
            "projects": {},
            "tokens": {},
            "chains": {},
            "topics": {},
            "people": {},
        }
        for bucket, values in entity_sources.items():
            for entity, sources in values.items():
                normalized_entity = self._normalize_topics([entity])[0] if bucket == "topics" else entity
                normalized.setdefault(bucket, {}).setdefault(normalized_entity, [])
                for source in sources:
                    if source not in normalized[bucket][normalized_entity]:
                        normalized[bucket][normalized_entity].append(source)
        return normalized

    def _build_entity_confidence(
        self,
        entity_sources: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, Dict[str, str]]:
        confidence: Dict[str, Dict[str, str]] = {
            "projects": {},
            "tokens": {},
            "chains": {},
            "topics": {},
            "people": {},
        }
        for bucket, values in entity_sources.items():
            for entity, sources in values.items():
                has_tag = any(source.startswith("tag:") for source in sources)
                has_title = any(source.startswith("title:") for source in sources)
                has_body = any(source.startswith("body:") for source in sources)
                if has_tag and (has_title or has_body):
                    level = "strong"
                elif has_title and has_body:
                    level = "strong"
                elif has_title:
                    level = "medium"
                elif bucket in {"chains", "topics"} and has_body:
                    level = "strong"
                elif has_tag:
                    level = "medium"
                else:
                    level = "weak"
                confidence.setdefault(bucket, {})[entity] = level
        return confidence

    def _count_confidence_level(self, values: Dict[str, str], target_level: str) -> int:
        return sum(1 for level in values.values() if level == target_level)

    def _has_title_entity_hit(self, entity_sources: Dict[str, Dict[str, List[str]]]) -> bool:
        for bucket in ("projects", "tokens", "chains", "topics"):
            for sources in entity_sources.get(bucket, {}).values():
                if any(source.startswith("title:") for source in sources):
                    return True
        return False

    def _has_crypto_entity_signal(self, item: ContentItem) -> bool:
        entity_confidence = item.entity_confidence
        for bucket in ("projects", "tokens", "chains", "topics"):
            values = entity_confidence.get(bucket, {})
            if any(level in {"strong", "medium"} for level in values.values()):
                return True
        return bool(
            item.entities.projects
            or item.entities.tokens
            or item.entities.chains
            or item.entities.topics
        )

    def _build_event_type_scores(
        self,
        item: ContentItem,
        text: str,
        title_text: str,
        is_crypto: bool,
        has_macro_signal: bool,
        has_noise_signal: bool,
    ) -> Dict[EventType, float]:
        scores = self._init_event_type_scores()
        self._score_event_type_from_entities(scores, item)
        self._score_event_type_from_text(scores, text, title_text)
        self._boost_event_type_scores_with_entities(scores, item)

        if is_crypto and any(self._contains_alias(text, keyword) for keyword in INSTITUTIONAL_KEYWORDS):
            if any(self._contains_alias(text, name) for name in INSTITUTION_NAMES):
                scores[EventType.INSTITUTIONAL_ADOPTION] += 0.70

        if is_crypto and any(self._contains_alias(text, keyword) for keyword in LAUNCH_KEYWORDS):
            scores[EventType.PRODUCT_LAUNCH] += 0.55

        if has_macro_signal:
            scores[EventType.MACRO] += 0.55 if not is_crypto else 0.25

        if has_noise_signal and not is_crypto:
            scores[EventType.MACRO] += 0.45

        if any(self._contains_alias(title_text, keyword) for keyword in TITLE_MARKET_STRUCTURE_HINTS):
            scores[EventType.MARKET_STRUCTURE] += 0.25

        return scores

    def _init_event_type_scores(self) -> Dict[EventType, float]:
        event_types = {
            event_type
            for event_type, _ in EVENT_TYPE_RULES
        }
        event_types.update(
            {
                EventType.INSTITUTIONAL_ADOPTION,
                EventType.PRODUCT_LAUNCH,
                EventType.MACRO,
            }
        )
        return {event_type: 0.0 for event_type in event_types}

    def _score_event_type_from_entities(self, scores: Dict[EventType, float], item: ContentItem) -> None:
        topic_confidence = item.entity_confidence.get("topics", {})
        for topic in item.entities.topics:
            mapped_event_type = TOPIC_EVENT_TYPE_MAP.get(topic)
            if not mapped_event_type:
                continue
            confidence = topic_confidence.get(topic, "weak")
            sources = item.entity_sources.get("topics", {}).get(topic, [])
            if confidence == "strong":
                scores[mapped_event_type] += 0.90
                if any(source.startswith("title:") for source in sources):
                    scores[mapped_event_type] += 0.10
            elif confidence == "medium":
                scores[mapped_event_type] += 0.45
                if any(source.startswith("title:") for source in sources):
                    scores[mapped_event_type] += 0.15
            elif any(source.startswith("title:") for source in sources):
                scores[mapped_event_type] += 0.10

    def _score_event_type_from_text(
        self,
        scores: Dict[EventType, float],
        text: str,
        title_text: str,
    ) -> None:
        for event_type, keywords in EVENT_TYPE_RULES:
            for keyword in keywords:
                if self._contains_alias(title_text, keyword):
                    scores[event_type] += 0.35
                elif self._contains_alias(text, keyword):
                    scores[event_type] += 0.20

    def _boost_event_type_scores_with_entities(
        self,
        scores: Dict[EventType, float],
        item: ContentItem,
    ) -> None:
        has_projects = bool(item.entities.projects)
        has_tokens = bool(item.entities.tokens)
        has_chains = bool(item.entities.chains)

        if has_tokens:
            if scores[EventType.LISTING] > 0:
                scores[EventType.LISTING] += 0.10
            if scores[EventType.TOKEN_UNLOCK] > 0:
                scores[EventType.TOKEN_UNLOCK] += 0.12

        if has_projects:
            if scores[EventType.FUNDING] > 0:
                scores[EventType.FUNDING] += 0.10
            if scores[EventType.PARTNERSHIP] > 0:
                scores[EventType.PARTNERSHIP] += 0.08
            if scores[EventType.MERGER] > 0:
                scores[EventType.MERGER] += 0.08
            if scores[EventType.PRODUCT_LAUNCH] > 0:
                scores[EventType.PRODUCT_LAUNCH] += 0.10
            if scores[EventType.GOVERNANCE] > 0:
                scores[EventType.GOVERNANCE] += 0.08
            if scores[EventType.INCENTIVE_PROGRAM] > 0:
                scores[EventType.INCENTIVE_PROGRAM] += 0.08

        if has_tokens or has_chains:
            if scores[EventType.MARKET_STRUCTURE] > 0:
                scores[EventType.MARKET_STRUCTURE] += 0.08

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(value, max_value))

    def _load_dynamic_aliases(self) -> None:
        config_path = Path(self.dynamic_alias_path)
        if not config_path.exists():
            return

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self._merge_alias_map(self.entity_aliases, payload.get("entity_aliases", {}))
        self._merge_alias_map(self.token_aliases, payload.get("token_aliases", {}))
        self._merge_alias_map(self.topic_aliases, payload.get("topic_aliases", {}))

    def _merge_alias_map(self, target: Dict[str, List[str]], updates: Dict[str, List[str]]) -> None:
        for canonical_name, aliases in updates.items():
            merged = list(target.get(canonical_name, []))
            for alias in aliases:
                normalized_alias = alias.strip()
                if normalized_alias and normalized_alias not in merged:
                    merged.append(normalized_alias)
            if merged:
                target[canonical_name] = merged
