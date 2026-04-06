from __future__ import annotations

from .event_types import EventType

DEFAULT_DYNAMIC_ALIAS_PATH = "data/candidate_discovery/promoted_aliases.json"

CHAIN_KEYWORDS = [
    "solana",
    "base",
    "ethereum",
    "bitcoin",
    "arbitrum",
    "optimism",
    "polygon",
    "bnb chain",
    "avalanche",
]

TOPIC_KEYWORDS = [
    "defi",
    "ai agents",
    "airdrop",
    "governance",
    "exploit",
    "listing",
    "funding",
    "memecoin",
    "token unlock",
    "partnership",
    "merger",
    "incentive program",
    "restaking",
    "stablecoin",
    "rwa",
    "derivatives",
    "layer2",
]

TITLE_EVENT_HINTS = {
    EventType.EXPLOIT: 0.18,
    EventType.GOVERNANCE: 0.12,
    EventType.PRODUCT_LAUNCH: 0.12,
    EventType.INSTITUTIONAL_ADOPTION: 0.12,
    EventType.LISTING: 0.10,
    EventType.FUNDING: 0.10,
    EventType.TOKEN_UNLOCK: 0.10,
    EventType.INCENTIVE_PROGRAM: 0.09,
    EventType.MARKET_STRUCTURE: 0.08,
    EventType.AIRDROP: 0.10,
    EventType.PARTNERSHIP: 0.08,
    EventType.MERGER: 0.08,
    EventType.MACRO: 0.08,
}

TOKEN_ALIASES = {
    "BTC": ["btc", "bitcoin"],
    "ETH": ["eth", "ethereum"],
    "SOL": ["sol", "solana"],
    "JUP": ["jup", "jupiter"],
    "HYPE": ["hype", "hyperliquid"],
    "ARB": ["arb", "arbitrum"],
    "OP": ["op", "optimism"],
    "BNB": ["bnb", "binance coin", "bnb chain"],
    "AVAX": ["avax", "avalanche"],
    "WIF": ["wif", "dogwifhat"],
    "BONK": ["bonk"],
    "ENA": ["ena", "ethena"],
    "ONDO": ["ondo"],
    "MKR": ["mkr", "maker"],
    "AAVE": ["aave"],
}

ENTITY_ALIASES = {
    "Bitcoin": ["bitcoin", "btc"],
    "Ethereum": ["ethereum", "eth"],
    "Solana": ["solana", "sol"],
    "Base": ["base", "base chain", "coinbase l2"],
    "Jupiter": ["jupiter", "jup"],
    "Hyperliquid": ["hyperliquid", "hype"],
    "Arbitrum": ["arbitrum", "arb"],
    "Optimism": ["optimism", "op mainnet"],
    "Polygon": ["polygon", "matic", "polygon pos"],
    "BNB Chain": ["bnb chain", "binance smart chain", "bsc"],
    "Avalanche": ["avalanche", "avax"],
    "Ethena": ["ethena", "ena"],
    "Ondo": ["ondo", "ondo finance"],
    "Aave": ["aave"],
    "Maker": ["maker", "makerdao", "sky"],
    "Drift": ["drift"],
    "Kamino": ["kamino"],
    "Raydium": ["raydium", "ray"],
    "Pump.fun": ["pump.fun", "pumpfun"],
    "dogwifhat": ["dogwifhat", "wif"],
    "Bonk": ["bonk"],
    "Kraken": ["kraken"],
    "Coinbase": ["coinbase"],
    "Binance": ["binance"],
}

TOPIC_ALIASES = {
    "DeFi": ["defi", "dex", "amm", "yield", "lending", "perp", "perpetual"],
    "AI Agents": ["ai agents", "ai agent", "agent", "autonomous agent"],
    "NFT": ["nft", "ordinal", "ordinals"],
    "Governance": ["governance", "proposal", "vote", "snapshot"],
    "Airdrop": ["airdrop", "claim", "retroactive"],
    "Exploit": ["exploit", "hack", "drain", "breach"],
    "Listing": [
        "listing",
        "listed",
        "open trading",
        "trading starts",
        "上线交易",
        "上架交易",
        "登陆交易所",
    ],
    "Funding": [
        "raise",
        "funding",
        "financing",
        "seed round",
        "series a",
        "series b",
        "strategic round",
        "融资",
        "募资",
    ],
    "Memecoin": ["memecoin", "meme coin", "meme", "dog token"],
    "Token Unlock": ["token unlock", "unlocks", "vesting", "cliff", "解锁", "归属"],
    "Partnership": ["partnership", "collaboration", "partnered", "合作", "integrates with", "集成"],
    "Merger": ["acquire", "acquisition", "merge", "merger", "收购", "合并"],
    "Incentive Program": [
        "reward program",
        "points program",
        "launchpool",
        "liquidity incentive",
        "mining campaign",
        "积分计划",
        "激励计划",
        "流动性激励",
    ],
    "Restaking": ["restaking", "liquid restaking", "lrt", "eigenlayer"],
    "Stablecoin": ["stablecoin", "stables", "usdt", "usdc", "susde"],
    "RWA": ["rwa", "real world asset", "tokenized treasury", "treasury tokenization"],
    "Derivatives": ["derivatives", "futures", "options", "perps", "open interest"],
    "Layer2": ["layer2", "layer 2", "l2", "rollup", "optimistic rollup"],
}

MACRO_KEYWORDS = [
    "fed", "fomc", "powell", "cpi", "ppi", "inflation", "rate cut", "rate hike",
    "nonfarm", "payroll", "tariff", "trade deficit", "treasury", "bond yield",
    "liquidity", "recession", "risk-on", "risk-off", "美联储", "非农", "通胀",
    "加息", "降息", "关税", "贸易逆差", "国债收益率", "流动性", "衰退",
]

NON_CRYPTO_NOISE_KEYWORDS = [
    "大使馆", "以色列", "伊朗", "沙特", "无人机", "空袭", "军事", "战争", "袭击",
    "embassy", "israel", "iran", "saudi", "missile", "drone", "airstrike", "military",
]

MARKET_STRUCTURE_KEYWORDS = [
    "open interest",
    "liquidation",
    "short squeeze",
    "etf flow",
    "whale transfer",
    "funding rate",
    "清算",
    "爆仓",
    "鲸鱼转账",
    "资金费率",
]

EVENT_TYPE_RULES = [
    (
        EventType.EXPLOIT,
        ["exploit", "hack", "drain", "breach", "攻击", "被盗", "漏洞", "security incident"],
    ),
    (
        EventType.GOVERNANCE,
        ["proposal", "vote", "governance", "提案", "投票", "治理", "snapshot"],
    ),
    (
        EventType.LISTING,
        [
            "listing",
            "listed",
            "open trading",
            "trading starts",
            "上线交易",
            "上架交易",
            "登陆币安",
            "登陆交易所",
        ],
    ),
    (
        EventType.FUNDING,
        [
            "raise",
            "funding",
            "financing",
            "seed round",
            "series a",
            "series b",
            "strategic round",
            "融资",
            "募资",
        ],
    ),
    (
        EventType.TOKEN_UNLOCK,
        [
            "token unlock",
            "unlocks",
            "vesting",
            "cliff",
            "解锁",
            "归属",
        ],
    ),
    (
        EventType.AIRDROP,
        ["airdrop", "claim", "retroactive", "空投", "领取", "申领"],
    ),
    (
        EventType.PARTNERSHIP,
        ["partnership", "collaboration", "partnered", "合作", "integrates with", "集成"],
    ),
    (
        EventType.MERGER,
        ["acquire", "acquisition", "merge", "merger", "收购", "合并"],
    ),
    (
        EventType.INCENTIVE_PROGRAM,
        [
            "reward program",
            "points program",
            "launchpool",
            "liquidity incentive",
            "mining campaign",
            "积分计划",
            "激励计划",
            "流动性激励",
        ],
    ),
    (EventType.MARKET_STRUCTURE, MARKET_STRUCTURE_KEYWORDS),
]

INSTITUTIONAL_KEYWORDS = [
    "现货交易", "spot trading", "提供交易", "支持交易", "开放交易",
    "计划推出", "plans to launch", "considering offering", "考虑推出", "考虑提供",
]

INSTITUTION_NAMES = [
    "嘉信理财", "schwab", "fidelity", "blackrock", "coinbase", "binance", "okx",
]

LAUNCH_KEYWORDS = [
    "launch", "announced", "release", "mainnet", "推出", "主网上线", "goes live", "live now",
]

TITLE_MARKET_STRUCTURE_HINTS = ["surge", "rally", "跌破", "突破"]

TOPIC_EVENT_TYPE_MAP = {
    "Exploit": EventType.EXPLOIT,
    "Governance": EventType.GOVERNANCE,
    "Airdrop": EventType.AIRDROP,
    "Listing": EventType.LISTING,
    "Funding": EventType.FUNDING,
    "Token Unlock": EventType.TOKEN_UNLOCK,
    "Partnership": EventType.PARTNERSHIP,
    "Merger": EventType.MERGER,
    "Incentive Program": EventType.INCENTIVE_PROGRAM,
    "Derivatives": EventType.MARKET_STRUCTURE,
}
