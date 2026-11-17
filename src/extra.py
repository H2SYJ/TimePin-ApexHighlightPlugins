"""Optional HUD info enrichment: rank, hero, player names, and combat statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RankInfo:
    tier: str | None = None
    name: str | None = None
    division: str | None = None
    confidence: float | None = None
    method: str | None = None
    mode: str | None = None


@dataclass
class HeroInfo:
    id: str | None = None
    name: str | None = None
    confidence: float | None = None
    method: str | None = None


@dataclass
class PlayerNameInfo:
    name: str | None = None
    confidence: float | None = None
    method: str | None = None


@dataclass
class TeammateInfo:
    name: str | None = None
    confidence: float | None = None
    method: str | None = None
    slot: int | None = None


@dataclass
class CombatStats:
    damage: int | None = None
    kills: int | None = None
    assists: int | None = None
    hits: int | None = None
    duration: float | None = None


@dataclass
class HUDInfo:
    rank: RankInfo | None = None
    hero: HeroInfo | None = None
    player_name: PlayerNameInfo | None = None
    teammates: list[TeammateInfo] | None = None
    stats: CombatStats | None = None


def create_empty_hud_info() -> HUDInfo:
    """Create an empty HUDInfo with all fields set to None."""
    return HUDInfo(
        rank=RankInfo(),
        hero=HeroInfo(),
        player_name=PlayerNameInfo(),
        teammates=[],
        stats=CombatStats(),
    )


def hud_info_to_metadata(hud_info: HUDInfo) -> dict[str, str]:
    """Convert HUDInfo to Mark.metadata dictionary (string values only)."""
    metadata: dict[str, str] = {}

    if hud_info.rank and hud_info.rank.tier:
        metadata["rank_tier"] = hud_info.rank.tier
        if hud_info.rank.name:
            metadata["rank_name"] = hud_info.rank.name
        if hud_info.rank.division:
            metadata["rank_division"] = hud_info.rank.division
        if hud_info.rank.confidence is not None:
            metadata["rank_confidence"] = str(hud_info.rank.confidence)
        if hud_info.rank.method:
            metadata["rank_method"] = hud_info.rank.method
        if hud_info.rank.mode:
            metadata["rank_mode"] = hud_info.rank.mode

    if hud_info.hero and hud_info.hero.id:
        metadata["hero_id"] = hud_info.hero.id
        if hud_info.hero.name:
            metadata["hero_name"] = hud_info.hero.name
        if hud_info.hero.confidence is not None:
            metadata["hero_confidence"] = str(hud_info.hero.confidence)
        if hud_info.hero.method:
            metadata["hero_method"] = hud_info.hero.method

    if hud_info.player_name and hud_info.player_name.name:
        metadata["player_name"] = hud_info.player_name.name
        if hud_info.player_name.confidence is not None:
            metadata["player_name_confidence"] = str(hud_info.player_name.confidence)
        if hud_info.player_name.method:
            metadata["player_name_method"] = hud_info.player_name.method

    if hud_info.teammates:
        teammate_names = []
        for i, tm in enumerate(hud_info.teammates):
            if tm.name:
                teammate_names.append(tm.name)
        if teammate_names:
            metadata["teammates"] = ",".join(teammate_names)

    if hud_info.stats:
        if hud_info.stats.damage is not None:
            metadata["stats_damage"] = str(hud_info.stats.damage)
        if hud_info.stats.kills is not None:
            metadata["stats_kills"] = str(hud_info.stats.kills)
        if hud_info.stats.assists is not None:
            metadata["stats_assists"] = str(hud_info.stats.assists)
        if hud_info.stats.hits is not None:
            metadata["stats_hits"] = str(hud_info.stats.hits)
        if hud_info.stats.duration is not None:
            metadata["stats_duration"] = str(hud_info.stats.duration)

    return metadata


def get_resources_dir() -> Path:
    """Get the resources directory path."""
    return Path(__file__).resolve().parent.parent / "resources"


RANK_TEMPLATES_DIR = get_resources_dir() / "ranks"
LEGENDS_DIR = get_resources_dir() / "legends"