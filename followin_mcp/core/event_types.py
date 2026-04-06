from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    EXPLOIT = "exploit"
    GOVERNANCE = "governance"
    LISTING = "listing"
    FUNDING = "funding"
    TOKEN_UNLOCK = "token_unlock"
    AIRDROP = "airdrop"
    PARTNERSHIP = "partnership"
    MERGER = "merger"
    INCENTIVE_PROGRAM = "incentive_program"
    MARKET_STRUCTURE = "market_structure"
    INSTITUTIONAL_ADOPTION = "institutional_adoption"
    PRODUCT_LAUNCH = "product_launch"
    MACRO = "macro"
    UNKNOWN = "unknown"
