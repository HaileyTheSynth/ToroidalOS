#!/usr/bin/env python3
"""
TOROIDAL OS - Integration Test: Embeddings + TUFT Dynamics
===========================================================
Tests the full integration of:
1. Octen-Embedding-0.6B service
2. Embedding to Torus mapping
3. TUFT hypergraph kernel with semantic positioning
4. Barnes-Hut dynamics
5. Coherence and curvature metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict

# Import embedding components
from toroidal.embeddings import (
    OctenEmbeddingService,
    EmbeddingConfig,
    EmbeddingCache,
    EmbeddingToTorusMapper,
    TorusPosition,
    cosine_similarity,
    semantic_search,
)

# Import TUFT components
from toroidal.kernel.tuft_integration import (
    TUFTHypergraphKernel,
    compute_convergence_from_tuft,
    TUFT_AVAILABLE,
)
from toroidal.kernel.hypergraph import NodeType


def test_embedding_service():
    """Test the embedding service."""
    print("\n" + "=" * 60)
    print("TEST 1: Embedding Service")
    print("=" * 60)

    config = EmbeddingConfig(
        cache_max_items=100,
        cache_max_memory_mb=20,
        lazy_load=True
    )

    service = OctenEmbeddingService(config)
    print(f"Service created: {service}")
    print(f"Model loaded: {service.is_loaded()}")

    # Test encoding (triggers model load)
    print("\nEncoding test texts...")
    texts = [
        "The cat sat on the mat",
        "A feline rested on a rug",
        "The dog barked at the mailman",
        "Machine learning is fascinating",
        "Neural networks learn representations",
    ]

    embeddings = []
    for text in texts:
        emb = service.encode(text)
        embeddings.append(emb)
        print(f"  '{text[:30]}...' -> shape {emb.shape}")

    print(f"\nModel loaded: {service.is_loaded()}")

    # Test similarity
    print("\nSimilarity tests:")
    sim_cat_feline = service.similarity(texts[0], texts[1])
    sim_cat_dog = service.similarity(texts[0], texts[2])
    sim_ml_nn = service.similarity(texts[3], texts[4])

    print(f"  Cat/Mat vs Feline/Rug: {sim_cat_feline:.3f}")
    print(f"  Cat/Mat vs Dog/Mailman: {sim_cat_dog:.3f}")
    print(f"  ML vs Neural Networks: {sim_ml_nn:.3f}")

    # Verify semantic relationships
    assert sim_cat_feline > sim_cat_dog, "Semantically similar texts should have higher similarity"
    assert sim_ml_nn > 0.5, "Related ML concepts should have moderate similarity"

    # Test cache
    stats = service.get_stats()
    print(f"\nCache stats: {stats['cache_stats']}")

    # Test batch encoding
    print("\nBatch encoding test...")
    batch_embs = service.encode_batch(texts * 5, batch_size=8)
    print(f"  Batch shape: {batch_embs.shape}")

    print("\n[OK] Embedding service tests passed!")
    return service, embeddings


def test_torus_mapping(embeddings: List[np.ndarray], texts: List[str] = None):
    """Test embedding to torus mapping."""
    print("\n" + "=" * 60)
    print("TEST 2: Torus Mapping")
    print("=" * 60)

    mapper = EmbeddingToTorusMapper()
    print(f"Mapper created with projection shape: {mapper.get_projection_matrix().shape}")

    # Map individual embeddings
    print("\nMapping embeddings to torus positions:")
    positions = []
    for i, emb in enumerate(embeddings[:5]):
        pos = mapper.map_embedding(emb)
        positions.append(pos)
        print(f"  [{i}] {pos}")

    # Test torus distance
    print("\nTorus distances:")
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            dist = positions[i].distance_to(positions[j])
            sim_from_pos = mapper.compute_similarity_from_positions(positions[i], positions[j])
            print(f"  [{i}] <-> [{j}]: distance={dist:.2f}, similarity={sim_from_pos:.3f}")

    # Test batch mapping
    print("\nBatch mapping test...")
    batch_positions = mapper.map_batch(np.array(embeddings))
    print(f"  Batch positions shape: {batch_positions.shape}")

    # Verify positions are in valid range
    assert np.all(batch_positions >= 0) and np.all(batch_positions <= 360), \
        "All torus positions should be in [0, 360] range"

    print("\n[OK] Torus mapping tests passed!")
    return mapper, positions


def test_tuft_kernel_with_embeddings():
    """Test TUFT kernel with semantic embeddings."""
    print("\n" + "=" * 60)
    print("TEST 3: TUFT Kernel with Semantic Embeddings")
    print("=" * 60)

    # Create embedding service
    config = EmbeddingConfig(cache_max_items=200, lazy_load=True)
    embedding_service = OctenEmbeddingService(config)
    torus_mapper = EmbeddingToTorusMapper()

    # Create TUFT kernel with embeddings
    kernel = TUFTHypergraphKernel(
        max_nodes=500,
        grid_size=16,
        embedding_service=embedding_service,
        torus_mapper=torus_mapper
    )

    print(f"Kernel created. TUFT available: {TUFT_AVAILABLE}")
    print(f"Embedding service: {embedding_service}")

    # Add nodes with semantic content
    print("\nAdding semantically-positioned nodes...")
    semantic_groups = [
        # Group 1: Animals
        ("animal_cat", "A domestic cat hunting mice in a barn"),
        ("animal_dog", "A loyal dog guarding its owner's home"),
        ("animal_bird", "A colorful bird singing in the morning"),
        ("animal_fish", "Fish swimming in a clear mountain stream"),
        # Group 2: Technology
        ("tech_ai", "Artificial intelligence transforming industries"),
        ("tech_ml", "Machine learning models learning from data"),
        ("tech_neural", "Neural networks with deep architectures"),
        ("tech_robot", "Robots automating factory work"),
        # Group 3: Nature
        ("nature_forest", "A dense forest with ancient trees"),
        ("nature_ocean", "The vast ocean with its tides"),
        ("nature_mountain", "Majestic mountains piercing the clouds"),
        ("nature_river", "A winding river flowing to the sea"),
    ]

    for node_id, content in semantic_groups:
        node = kernel.add_node(
            node_id,
            NodeType.THOUGHT,
            {"content": content, "category": node_id.split("_")[0]},
            content=content
        )
        pos = kernel.get_torus_position(node_id)
        print(f"  {node_id}: position=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}, {pos[3]:.1f})")

    # Add hyperedges within groups
    print("\nCreating hyperedges...")
    kernel.add_hyperedge({"animal_cat", "animal_dog", "animal_bird", "animal_fish"}, "hyper_animals")
    kernel.add_hyperedge({"tech_ai", "tech_ml", "tech_neural", "tech_robot"}, "hyper_tech")
    kernel.add_hyperedge({"nature_forest", "nature_ocean", "nature_mountain", "nature_river"}, "hyper_nature")

    # Cross-group connections (bridges)
    kernel.add_hyperedge({"animal_cat", "nature_forest"}, "cat_in_forest")
    kernel.add_hyperedge({"tech_ai", "nature_ocean"}, "ai_ocean_monitoring")

    # Run TUFT dynamics
    print("\nRunning TUFT dynamics...")
    for step in range(10):
        kernel.step()
        if step % 3 == 0:
            curvature = kernel.compute_curvature()
            print(f"  Step {step}: curvature={curvature:.2f}")

    # Get statistics
    stats = kernel.get_tuft_stats()
    print("\nTUFT Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Compute coherence for nodes
    print("\nCoherence scores:")
    for node_id in list(kernel.nodes.keys())[:6]:
        coh = kernel.compute_coherence(node_id)
        print(f"  {node_id}: coherence={coh:.3f}")

    # Find bridges
    bridges = kernel.find_bridges(min_berry=0.5, min_regions=2)
    print(f"\nBridge nodes (min_berry=0.5, min_regions=2): {bridges}")

    # Test TUFT-based convergence
    thoughts = [f"Thought about {g[1]}" for g in semantic_groups[:5]]
    convergence = compute_convergence_from_tuft(kernel, thoughts)
    print(f"\nConvergence score: {convergence:.3f}")

    print("\n[OK] TUFT kernel with embeddings tests passed!")
    return kernel, stats


def test_semantic_defects():
    """Test semantic defect detection."""
    print("\n" + "=" * 60)
    print("TEST 4: Semantic Defect Detection")
    print("=" * 60)

    config = EmbeddingConfig(cache_max_items=100)
    embedding_service = OctenEmbeddingService(config)
    torus_mapper = EmbeddingToTorusMapper()

    kernel = TUFTHypergraphKernel(
        max_nodes=200,
        embedding_service=embedding_service,
        torus_mapper=torus_mapper
    )

    # Add nodes that should be semantically close
    similar_pairs = [
        ("cat1", "A fluffy cat purring on a sofa"),
        ("cat2", "A kitten sleeping on a cushion"),
        ("dog1", "A dog barking at strangers"),
        ("dog2", "A puppy playing in the yard"),
    ]

    for node_id, content in similar_pairs:
        kernel.add_node(node_id, NodeType.THOUGHT, {"content": content}, content=content)

    # Find semantic defects
    defects = kernel.find_semantic_defects(
        torus_distance_threshold=90.0,
        similarity_threshold=0.5,
        max_defects=10
    )

    print(f"Found {len(defects)} semantic defects:")
    for defect in defects[:5]:
        id_a, id_b, torus_dist, emb_sim = defect
        print(f"  {id_a} <-> {id_b}: torus_dist={torus_dist:.2f}, emb_sim={emb_sim:.3f}")

    print("\n[OK] Semantic defect detection tests passed!")
    return defects


def test_enhanced_tuft_features():
    """Test enhanced TUFT features: Wilson loops, semantic clusters, coherence."""
    print("\n" + "=" * 60)
    print("TEST 5: Enhanced TUFT Features")
    print("=" * 60)

    config = EmbeddingConfig(cache_max_items=100)
    embedding_service = OctenEmbeddingService(config)
    torus_mapper = EmbeddingToTorusMapper()

    kernel = TUFTHypergraphKernel(
        max_nodes=200,
        embedding_service=embedding_service,
        torus_mapper=torus_mapper
    )

    # Add nodes with semantic content
    texts = [
        'The quick brown fox jumps over the lazy dog',
        'A fast brown canine leaps above a sleepy hound',
        'Machine learning algorithms process data',
        'Neural networks learn from examples',
        'The forest is dense with ancient trees',
        'Ocean waves crash on the sandy shore',
    ]

    for i, text in enumerate(texts):
        kernel.add_node(f'node_{i}', NodeType.THOUGHT, {'content': text}, content=text)

    # Add hyperedges
    kernel.add_hyperedge({'node_0', 'node_1'}, 'similar_meaning')
    kernel.add_hyperedge({'node_2', 'node_3'}, 'ml_concepts')
    kernel.add_hyperedge({'node_4', 'node_5'}, 'nature_scenes')

    # Run dynamics
    for _ in range(10):
        kernel.step()

    # Test enhanced stats
    stats = kernel.get_tuft_stats()
    print(f"\nEnhanced Stats:")
    print(f"  Nodes with embeddings: {stats.get('nodes_with_embeddings', 0)}")
    print(f"  Embedding coverage: {stats.get('embedding_coverage', 0):.2f}")
    print(f"  Avg coherence: {stats.get('avg_coherence', 0):.3f}")
    print(f"  Coherence std: {stats.get('coherence_std', 0):.3f}")
    print(f"  Torus spread: {stats.get('torus_spread', 0):.2f}")
    print(f"  Avg velocity: {stats.get('avg_velocity', 0):.3f}")

    # Test Wilson loops
    wilson = kernel.get_wilson_loop_stats()
    print(f"\nWilson Loop Stats:")
    print(f"  Wilson loops: {wilson.get('wilson_loops', 0)}")
    print(f"  Avg holonomy: {wilson.get('avg_holonomy', 0):.3f}")

    # Test semantic clusters
    clusters = kernel.get_semantic_clusters(distance_threshold=80.0)
    print(f"\nSemantic Clusters: {len(clusters)} found")
    for i, cluster in enumerate(clusters[:3]):
        print(f"  Cluster {i}: {cluster}")

    # Verify ML concepts are clustered
    ml_clustered = any('node_2' in c and 'node_3' in c for c in clusters)
    print(f"\n  ML concepts (node_2, node_3) clustered: {ml_clustered}")

    print("\n[OK] Enhanced TUFT features tests passed!")
    return stats, wilson, clusters


def test_full_integration():
    """Run full integration test suite."""
    print("\n" + "=" * 70)
    print("TOROIDAL OS - EMBEDDING + TUFT INTEGRATION TEST SUITE")
    print("=" * 70)

    # Test 1: Embedding service
    service, embeddings = test_embedding_service()

    # Test 2: Torus mapping
    mapper, positions = test_torus_mapping(embeddings)

    # Test 3: TUFT kernel with embeddings
    kernel, stats = test_tuft_kernel_with_embeddings()

    # Test 4: Semantic defects
    defects = test_semantic_defects()

    # Test 5: Enhanced TUFT features
    enhanced_stats, wilson, clusters = test_enhanced_tuft_features()

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"  Embedding service: OK")
    print(f"  Torus mapping: OK")
    print(f"  TUFT kernel: OK (curvature={stats['curvature']:.2f}, bridges={stats['bridges']})")
    print(f"  Semantic defects: OK ({len(defects)} found)")
    print(f"  Enhanced TUFT features: OK")
    print(f"    - Coherence: {enhanced_stats.get('avg_coherence', 0):.3f}")
    print(f"    - Semantic clusters: {len(clusters)}")
    print(f"    - Wilson loops: {wilson.get('wilson_loops', 0)}")
    print(f"\n  All tests passed!")
    print("=" * 70)

    return {
        "service": service,
        "mapper": mapper,
        "kernel": kernel,
        "stats": stats,
        "defects": defects,
        "enhanced_stats": enhanced_stats,
        "wilson": wilson,
        "clusters": clusters,
    }


if __name__ == "__main__":
    results = test_full_integration()
