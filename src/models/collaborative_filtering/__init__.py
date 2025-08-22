"""Collaborative filtering models."""

from src.models.collaborative_filtering.user_based import UserBasedCF
from src.models.collaborative_filtering.item_based import ItemBasedCF

__all__ = ["UserBasedCF", "ItemBasedCF"]