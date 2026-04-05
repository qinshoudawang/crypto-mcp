from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class Entities:
    projects: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    chains: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)


@dataclass
class ContentItem:
    id: str
    title: str
    summary: str
    content: str
    url: str
    source_type: str
    source_name: str
    author: str
    published_at: datetime
    language: str

    entities: Entities = field(default_factory=Entities)
    event_type: str = "unknown"
    importance_score: float = 0.0
    credibility_score: float = 0.5
    engagement_score: float = 0.0
    raw_tags: List[str] = field(default_factory=list)
    entity_sources: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    entity_confidence: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class EventCluster:
    event_id: str
    title: str
    event_type: str
    entities: Entities
    first_seen_at: datetime
    last_updated_at: datetime
    importance_score: float

    items: List[ContentItem] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    user_id: str
    interests: List[str] = field(default_factory=list)
    muted_topics: List[str] = field(default_factory=list)
    preferred_languages: List[str] = field(default_factory=lambda: ["zh"])
    risk_preference: str = "medium"
    followed_projects: List[str] = field(default_factory=list)
    followed_kols: List[str] = field(default_factory=list)
