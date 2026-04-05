from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from .models import ContentItem, Entities


class ContentNormalizer:
    CHAIN_KEYWORDS = ["solana", "base", "ethereum", "bitcoin"]
    TOPIC_KEYWORDS = ["defi", "ai agents", "airdrop", "governance", "exploit"]
    TOKEN_ALIASES = {
        "BTC": ["btc", "bitcoin"],
        "ETH": ["eth", "ethereum"],
        "SOL": ["sol", "solana"],
        "JUP": ["jup", "jupiter"],
        "HYPE": ["hype", "hyperliquid"],
    }
    ENTITY_ALIASES = {
        "Bitcoin": ["bitcoin", "btc"],
        "Ethereum": ["ethereum", "eth"],
        "Solana": ["solana", "sol"],
        "Base": ["base", "base chain", "coinbase l2"],
        "Jupiter": ["jupiter", "jup"],
        "Hyperliquid": ["hyperliquid", "hype"],
    }
    TOPIC_ALIASES = {
        "DeFi": ["defi", "dex", "amm", "yield", "lending", "perp", "perpetual"],
        "AI Agents": ["ai agents", "ai agent", "agent", "autonomous agent"],
        "NFT": ["nft", "ordinal", "ordinals"],
        "Governance": ["governance", "proposal", "vote", "snapshot"],
        "Airdrop": ["airdrop", "claim", "retroactive"],
        "Exploit": ["exploit", "hack", "drain", "breach"],
    }
    NON_CRYPTO_KEYWORDS = [
        "大使馆", "以色列", "伊朗", "沙特", "无人机", "空袭", "军事", "战争", "袭击",
        "embassy", "israel", "iran", "saudi", "missile", "drone", "airstrike", "military",
    ]

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

        text = f"{item.title} {item.content}".lower()

        for chain in self.CHAIN_KEYWORDS:
            if self._contains_alias(text, chain):
                normalized_chain = chain.title()
                entities.chains.append(normalized_chain)
                self._record_entity_source(entity_sources, "chains", normalized_chain, f"text:{chain}")

        for topic in self.TOPIC_KEYWORDS:
            if self._contains_alias(text, topic):
                normalized_topic = self._title_case_topic(topic)
                entities.topics.append(normalized_topic)
                self._record_entity_source(entity_sources, "topics", normalized_topic, f"text:{topic}")

        for canonical_name, aliases in self.ENTITY_ALIASES.items():
            matched_aliases = [alias for alias in aliases if self._contains_alias(text, alias)]
            if matched_aliases:
                entities.projects.append(canonical_name)
                for alias in matched_aliases:
                    self._record_entity_source(entity_sources, "projects", canonical_name, f"text:{alias}")

        for symbol, aliases in self.TOKEN_ALIASES.items():
            matched_aliases = [alias for alias in aliases if self._contains_alias(text, alias)]
            if matched_aliases:
                entities.tokens.append(symbol)
                for alias in matched_aliases:
                    self._record_entity_source(entity_sources, "tokens", symbol, f"text:{alias}")

        for canonical_topic, aliases in self.TOPIC_ALIASES.items():
            matched_aliases = [alias for alias in aliases if self._contains_alias(text, alias)]
            if matched_aliases:
                entities.topics.append(canonical_topic)
                for alias in matched_aliases:
                    self._record_entity_source(entity_sources, "topics", canonical_topic, f"text:{alias}")

        entities.projects = self._dedupe_preserve_order(entities.projects)
        entities.tokens = self._dedupe_preserve_order(entities.tokens)
        entities.chains = self._dedupe_preserve_order(entities.chains)
        entities.topics = self._dedupe_preserve_order(self._normalize_topics(entities.topics))
        entities.people = list(dict.fromkeys(entities.people))
        item.entity_sources = self._normalize_entity_sources(entity_sources)
        item.entity_confidence = self._build_entity_confidence(item.entity_sources)
        return entities

    def classify_event_type(self, item: ContentItem) -> str:
        text = f"{item.title} {item.content}".lower()
        is_crypto = bool(item.entities.projects or item.entities.tokens or item.entities.chains or item.entities.topics)
        if any(keyword in text for keyword in self.NON_CRYPTO_KEYWORDS) and not is_crypto:
            return "macro"

        rules = {
            "exploit": ["exploit", "hack", "drain", "breach", "攻击", "被盗", "漏洞"],
            "partnership": ["partnership", "collaboration", "partnered", "合作"],
            "governance": ["proposal", "vote", "governance", "提案", "投票", "治理"],
            "airdrop": ["airdrop", "claim", "空投", "领取"],
            "listing": ["listing", "listed", "上线交易", "上架交易", "登陆币安", "登陆交易所"],
        }
        for event_type, keywords in rules.items():
            if any(self._contains_alias(text, keyword) for keyword in keywords):
                return event_type

        institutional_keywords = [
            "现货交易", "spot trading", "提供交易", "支持交易", "开放交易",
            "计划推出", "plans to launch", "considering offering", "考虑推出", "考虑提供",
        ]
        institution_names = [
            "嘉信理财", "schwab", "fidelity", "blackrock", "coinbase", "binance", "okx",
        ]
        if is_crypto and any(self._contains_alias(text, keyword) for keyword in institutional_keywords):
            if any(self._contains_alias(text, name) for name in institution_names):
                return "institutional_adoption"

        launch_keywords = ["launch", "announced", "program", "release", "推出", "主网上线"]
        if is_crypto and any(self._contains_alias(text, keyword) for keyword in launch_keywords):
            return "product_launch"
        return "unknown"

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
        score = 0.30
        if item.event_type in {"exploit", "governance", "product_launch", "institutional_adoption"}:
            score += 0.25
        if item.entities.projects:
            score += 0.15
        if item.entities.tokens:
            score += 0.10
        score += (item.credibility_score - 0.5) * 0.3
        return round(min(score, 1.0), 2)

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
                has_text = any(source.startswith("text:") for source in sources)
                if has_tag and has_text:
                    level = "strong"
                elif bucket in {"chains", "topics"} and has_text:
                    level = "strong"
                elif has_tag:
                    level = "medium"
                else:
                    level = "weak"
                confidence.setdefault(bucket, {})[entity] = level
        return confidence
