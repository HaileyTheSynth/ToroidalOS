#!/usr/bin/env python3
"""
TOROIDAL OS - Solenoid Memory
==============================
Multi-scale hierarchical memory with compression.

Inspired by TUFT's solenoid structure:
- Level 0: Raw recent experiences (seconds)
- Level 1: Compressed summaries (minutes)  
- Level 2: Abstracted patterns (hours)
- Level 3: Core beliefs/knowledge (persistent)

Each level "winds around" the previous, like the nested tori
in the solenoid construction from topology.
"""

import json
import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import threading


@dataclass
class MemoryItem:
    """A single memory item at any level"""
    id: str
    content: str
    level: int
    timestamp: float = field(default_factory=time.time)
    source_ids: List[str] = field(default_factory=list)  # What was compressed
    importance: float = 1.0
    access_count: int = 0
    embedding: Optional[List[float]] = None  # For semantic search
    
    def touch(self):
        self.access_count += 1
    
    def hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


class SolenoidLevel:
    """A single level in the solenoid memory hierarchy"""
    
    def __init__(self, level: int, max_items: int, name: str):
        self.level = level
        self.max_items = max_items
        self.name = name
        self.items: deque = deque(maxlen=max_items)
        self.index: Dict[str, MemoryItem] = {}  # id -> item
        
    def add(self, item: MemoryItem):
        """Add item to this level"""
        self.items.append(item)
        self.index[item.id] = item
        
        # Cleanup old index entries
        if len(self.index) > self.max_items * 2:
            current_ids = {i.id for i in self.items}
            self.index = {k: v for k, v in self.index.items() if k in current_ids}
    
    def get_recent(self, n: int = 10) -> List[MemoryItem]:
        """Get n most recent items"""
        return list(self.items)[-n:]
    
    def get_all(self) -> List[MemoryItem]:
        """Get all items at this level"""
        return list(self.items)
    
    def search(self, query: str) -> List[MemoryItem]:
        """Simple keyword search"""
        query_lower = query.lower()
        results = []
        for item in self.items:
            if query_lower in item.content.lower():
                item.touch()
                results.append(item)
        return results
    
    def is_full(self) -> bool:
        return len(self.items) >= self.max_items


class SolenoidMemory:
    """
    Multi-scale hierarchical memory system.
    
    Memory flows upward through compression:
    Raw → Summarized → Abstracted → Core
    
    This mirrors the nested torus structure of a solenoid,
    where each level wraps around and contains the essence
    of the level below.
    """
    
    def __init__(
        self,
        num_levels: int = 4,
        compression_ratio: int = 8,
        compressor: Callable = None
    ):
        """
        Initialize solenoid memory.
        
        Args:
            num_levels: Number of hierarchy levels
            compression_ratio: Items at level N compress to 1 at level N+1
            compressor: Function to compress items (uses LLM if provided)
        """
        self.num_levels = num_levels
        self.compression_ratio = compression_ratio
        self.compressor = compressor or self._default_compressor
        
        # Level sizes (smaller at higher levels)
        # For 6GB Mi Mix: ~50MB total for memory
        level_sizes = [
            64,   # Level 0: ~64 raw items (seconds)
            32,   # Level 1: ~32 summaries (minutes)
            16,   # Level 2: ~16 abstractions (hours)
            8     # Level 3: ~8 core beliefs (persistent)
        ]
        
        level_names = ["raw", "summary", "abstract", "core"]
        
        self.levels = []
        for i in range(num_levels):
            level = SolenoidLevel(
                level=i,
                max_items=level_sizes[i] if i < len(level_sizes) else 8,
                name=level_names[i] if i < len(level_names) else f"level_{i}"
            )
            self.levels.append(level)
        
        self._lock = threading.Lock()
        self._item_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique memory ID"""
        self._item_counter += 1
        return f"mem_{int(time.time())}_{self._item_counter}"
    
    def _default_compressor(self, items: List[MemoryItem]) -> str:
        """Default compression: concatenate and truncate"""
        contents = [item.content for item in items]
        combined = " | ".join(contents)
        # Truncate to ~500 chars
        if len(combined) > 500:
            combined = combined[:497] + "..."
        return f"[Compressed {len(items)} items]: {combined}"
    
    def wind(self, content: str, importance: float = 1.0, level: int = 0) -> MemoryItem:
        """
        Add content to memory, compressing upward as needed.
        
        "Wind" refers to the winding number in topology - 
        each addition winds the memory tighter.
        """
        with self._lock:
            # Create new item at specified level
            item = MemoryItem(
                id=self._generate_id(),
                content=content,
                level=level,
                importance=importance
            )
            
            self.levels[level].add(item)
            
            # Check if compression is needed
            self._maybe_compress(level)
            
            return item
    
    def _maybe_compress(self, level: int):
        """Compress level if full, propagating upward"""
        if level >= self.num_levels - 1:
            return  # Can't compress top level
        
        current_level = self.levels[level]
        
        if len(current_level.items) >= self.compression_ratio:
            # Get items to compress
            items_to_compress = list(current_level.items)[:self.compression_ratio]
            
            # Compress using provided compressor
            compressed_content = self.compressor(items_to_compress)
            
            # Calculate combined importance
            avg_importance = sum(i.importance for i in items_to_compress) / len(items_to_compress)
            
            # Create compressed item at next level
            compressed_item = MemoryItem(
                id=self._generate_id(),
                content=compressed_content,
                level=level + 1,
                importance=avg_importance * 1.1,  # Slightly boost importance
                source_ids=[i.id for i in items_to_compress]
            )
            
            self.levels[level + 1].add(compressed_item)
            
            # Remove compressed items from current level
            for _ in range(self.compression_ratio):
                if current_level.items:
                    current_level.items.popleft()
            
            # Recursively check if next level needs compression
            self._maybe_compress(level + 1)
    
    def unwind(self, include_levels: List[int] = None) -> str:
        """
        Retrieve memory context from all levels.
        
        "Unwind" - read back the wound memories.
        Returns a formatted string suitable for LLM context.
        """
        if include_levels is None:
            include_levels = list(range(self.num_levels))
        
        context_parts = []
        
        for level_idx in include_levels:
            if level_idx >= len(self.levels):
                continue
            
            level = self.levels[level_idx]
            items = level.get_recent(5)  # Get recent items from each level
            
            if items:
                level_name = level.name.upper()
                context_parts.append(f"[{level_name} MEMORY]")
                
                for item in reversed(items):  # Oldest first
                    age = time.time() - item.timestamp
                    age_str = self._format_age(age)
                    context_parts.append(f"  ({age_str}) {item.content}")
        
        return "\n".join(context_parts)
    
    def _format_age(self, seconds: float) -> str:
        """Format age in human-readable form"""
        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds/60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h ago"
        else:
            return f"{int(seconds/86400)}d ago"
    
    def search(self, query: str, levels: List[int] = None) -> List[MemoryItem]:
        """Search across memory levels"""
        if levels is None:
            levels = list(range(self.num_levels))
        
        results = []
        for level_idx in levels:
            if level_idx < len(self.levels):
                results.extend(self.levels[level_idx].search(query))
        
        # Sort by relevance (simple: importance * recency)
        results.sort(
            key=lambda x: x.importance * (1.0 / (time.time() - x.timestamp + 1)),
            reverse=True
        )
        
        return results
    
    def get_core_beliefs(self) -> List[MemoryItem]:
        """Get items from the highest (core) level"""
        return self.levels[-1].get_all()
    
    def inject_belief(self, content: str, importance: float = 2.0):
        """Directly inject a core belief at the highest level"""
        item = MemoryItem(
            id=self._generate_id(),
            content=content,
            level=self.num_levels - 1,
            importance=importance
        )
        self.levels[-1].add(item)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_items": 0,
            "levels": []
        }
        
        for level in self.levels:
            level_stats = {
                "name": level.name,
                "level": level.level,
                "items": len(level.items),
                "max_items": level.max_items,
                "fill_ratio": len(level.items) / level.max_items
            }
            stats["levels"].append(level_stats)
            stats["total_items"] += len(level.items)
        
        return stats
    
    def to_dict(self) -> Dict:
        """Serialize memory to dictionary"""
        return {
            "num_levels": self.num_levels,
            "compression_ratio": self.compression_ratio,
            "levels": [
                {
                    "name": level.name,
                    "items": [
                        {
                            "id": item.id,
                            "content": item.content,
                            "timestamp": item.timestamp,
                            "importance": item.importance
                        }
                        for item in level.items
                    ]
                }
                for level in self.levels
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict, compressor: Callable = None) -> 'SolenoidMemory':
        """Deserialize memory from dictionary"""
        memory = cls(
            num_levels=data["num_levels"],
            compression_ratio=data["compression_ratio"],
            compressor=compressor
        )
        
        for level_idx, level_data in enumerate(data["levels"]):
            for item_data in level_data["items"]:
                item = MemoryItem(
                    id=item_data["id"],
                    content=item_data["content"],
                    level=level_idx,
                    timestamp=item_data["timestamp"],
                    importance=item_data["importance"]
                )
                memory.levels[level_idx].add(item)
        
        return memory


class LLMCompressor:
    """Use LLM to compress memory items semantically"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def __call__(self, items: List[MemoryItem]) -> str:
        """Compress items using LLM"""
        contents = [item.content for item in items]
        
        prompt = f"""Compress these {len(items)} memories into one concise summary.
Preserve the most important information and key insights.

MEMORIES:
{chr(10).join(f'- {c}' for c in contents)}

COMPRESSED SUMMARY (1-2 sentences):"""

        try:
            response = self.llm.complete(prompt, max_tokens=100)
            return response.strip()
        except Exception as e:
            # Fallback to simple concatenation
            return f"[{len(items)} items]: " + " | ".join(c[:50] for c in contents)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing SolenoidMemory...")
    
    memory = SolenoidMemory(num_levels=4, compression_ratio=4)
    
    # Inject some core beliefs
    memory.inject_belief("I am Toroidal OS, a self-referential operating system.")
    memory.inject_belief("My purpose is to assist and reason with the user.")
    
    # Add some raw memories
    test_inputs = [
        "User said: Hello",
        "User asked about the weather",
        "I responded with current temperature",
        "User thanked me",
        "User asked about AI",
        "I explained machine learning basics",
        "User seemed interested",
        "User asked for a recommendation",
        "I suggested a book on AI",
        "User added it to reading list",
        "User said goodbye",
        "Session ended",
    ]
    
    for i, content in enumerate(test_inputs):
        memory.wind(content, importance=1.0 + (i % 3) * 0.2)
        print(f"Added: {content[:30]}...")
    
    # Print stats
    print("\n" + "="*50)
    print("MEMORY STATS:")
    stats = memory.get_stats()
    for level in stats["levels"]:
        print(f"  {level['name']}: {level['items']}/{level['max_items']} ({level['fill_ratio']*100:.0f}%)")
    
    # Unwind memory
    print("\n" + "="*50)
    print("UNWOUND MEMORY:")
    print(memory.unwind())
    
    # Search
    print("\n" + "="*50)
    print("SEARCH for 'AI':")
    results = memory.search("AI")
    for r in results[:3]:
        print(f"  [{r.level}] {r.content}")
    
    print("\nTest complete!")
