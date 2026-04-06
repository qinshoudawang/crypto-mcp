from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import requests


class FollowinSourceAdapter(Protocol):
    def get_latest_headlines_page(
        self,
        limit: int = 20,
        last_cursor: Optional[str] = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> Dict[str, Any]:
        ...

    def get_latest_headlines(
        self,
        limit: int = 20,
        last_cursor: Optional[str] = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> List[Dict[str, Any]]:
        ...

    def get_trending_feeds(
        self,
        feed_type: str = "hot_news",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        ...

    def get_project_feed_page(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def get_project_opinions_page(
        self,
        symbol: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def get_project_opinions(
        self,
        symbol: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def get_trending_topics_page(
        self,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...

    def get_trending_topics(
        self,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def search_content(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        ...


class FollowinAPIError(Exception):
    pass


class FollowinAPIAdapter:
    BASE_URL = "https://api.followin.io"

    def __init__(
        self,
        api_key: str,
        lang: str = "zh-Hans",
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key
        self.lang = lang
        self.timeout = timeout
        self.session = requests.Session()

    def get_latest_headlines_page(
        self,
        limit: int = 20,
        last_cursor: Optional[str] = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> Dict[str, Any]:
        params = {
            "apikey": self.api_key,
            "lang": self.lang,
            "count": min(limit, 30),
            "no_tag": str(no_tag).lower(),
            "only_important": str(only_important).lower(),
        }
        if last_cursor:
            params["last_cursor"] = last_cursor

        data = self._get("/open/feed/news", params)
        response: Dict[str, Any] = {"items": data.get("list", [])}
        response.update(self._extract_page_meta(data))
        return response

    def get_latest_headlines(
        self,
        limit: int = 20,
        last_cursor: Optional[str] = None,
        no_tag: bool = False,
        only_important: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.get_latest_headlines_page(
            limit=limit,
            last_cursor=last_cursor,
            no_tag=no_tag,
            only_important=only_important,
        ).get("items", [])

    def get_trending_feeds(
        self,
        feed_type: str = "hot_news",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if feed_type not in {"hot_news", "pop_info"}:
            raise ValueError("feed_type must be 'hot_news' or 'pop_info'")

        params = {
            "apikey": self.api_key,
            "lang": self.lang,
            "type": feed_type,
            "count": min(limit, 30),
        }
        data = self._get("/open/feed/list/trending", params)
        return data.get("list", [])

    def get_project_feed(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_project_feed_page(
            symbol=symbol,
            feed_type=feed_type,
            limit=limit,
            cursor=cursor,
        ).get("items", [])

    def get_project_feed_page(
        self,
        symbol: str,
        feed_type: str = "tag_information_feed",
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        allowed = {
            "tag_discussion_feed",
            "tag_information_feed",
            "news",
            "key_events",
        }
        if feed_type not in allowed:
            raise ValueError(f"feed_type must be one of {allowed}")

        params = {
            "apikey": self.api_key,
            "symbol": symbol,
            "type": feed_type,
            "lang": self.lang,
            "count": min(limit, 30),
        }
        if cursor:
            params["cursor"] = cursor

        data = self._get("/open/feed/list/tag", params)
        response: Dict[str, Any] = {"items": data.get("list", [])}
        response.update(self._extract_page_meta(data))
        return response

    def get_project_opinions(
        self,
        symbol: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_project_opinions_page(
            symbol=symbol,
            limit=limit,
            cursor=cursor,
        ).get("items", [])

    def get_project_opinions_page(
        self,
        symbol: str,
        limit: int = 20,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "apikey": self.api_key,
            "symbol": symbol,
            "lang": self.lang,
            "count": min(limit, 30),
        }
        if cursor:
            params["cursor"] = cursor

        data = self._get("/open/feed/list/tag/opinions", params)
        response: Dict[str, Any] = {"items": data.get("list", [])}
        response.update(self._extract_page_meta(data))
        return response

    def get_trending_topics(
        self,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self.get_trending_topics_page(limit=limit, cursor=cursor).get("items", [])

    def get_trending_topics_page(
        self,
        limit: int = 10,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "apikey": self.api_key,
            "lang": self.lang,
            "count": min(limit, 10),
        }
        if cursor:
            params["cursor"] = cursor

        data = self._get("/open/trending_topic/ranks", params)
        result: List[Dict[str, Any]] = []
        for day_block in data.get("list", []):
            for topic in day_block.get("topics", []):
                topic["_day_start_ts"] = day_block.get("day_start_ts")
                result.append(topic)
        response: Dict[str, Any] = {"items": result}
        response.update(self._extract_page_meta(data))
        return response

    def search_content(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        items = self.get_latest_headlines(limit=30)
        q = query.lower().strip()
        return [
            item
            for item in items
            if q in (item.get("title") or "").lower()
            or q in (item.get("content") or "").lower()
            or q in (item.get("translated_title") or "").lower()
            or q in (item.get("translated_content") or "").lower()
        ][:limit]

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()

        payload = resp.json()
        if payload.get("code") != 2000:
            raise FollowinAPIError(
                f"Followin API error: code={payload.get('code')} msg={payload.get('msg')}"
            )
        return payload.get("data", {})

    @staticmethod
    def _extract_page_meta(data: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        for key in ("cursor", "next_cursor", "last_cursor", "has_more", "has_next"):
            if key in data and data.get(key) is not None:
                meta[key] = data.get(key)
        return meta
