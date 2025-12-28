use ruvector_nervous_system::plasticity::btsp::{BTSPAssociativeMemory, BTSPLayer};

#[test]
fn test_complete_one_shot_workflow() {
    // Simulates realistic vector database scenario
    let mut layer = BTSPLayer::new(128, 2000.0);

    // Add 5 different patterns
    let patterns = vec![
        (vec![1.0; 128], 0.9),
        (vec![0.8; 128], 0.7),
        (vec![0.5; 128], 0.5),
        (vec![0.3; 128], 0.3),
        (vec![0.1; 128], 0.1),
    ];

    for (pattern, target) in &patterns {
        layer.one_shot_associate(pattern, *target);
    }

    // Verify all patterns retained
    for (pattern, expected) in &patterns {
        let output = layer.forward(pattern);
        let error = (output - expected).abs();
        assert!(error < 0.2, "Pattern recall failed: expected {}, got {} (error: {})",
                expected, output, error);
    }
}

#[test]
fn test_associative_memory_with_embeddings() {
    // Simulate storing text embeddings
    let mut memory = BTSPAssociativeMemory::new(256, 128);

    // Store 10 key-value pairs
    for i in 0..10 {
        let key = vec![i as f32 / 10.0; 256];
        let value = vec![(9 - i) as f32 / 10.0; 128];
        memory.store_one_shot(&key, &value).unwrap();
    }

    // Retrieve all
    for i in 0..10 {
        let key = vec![i as f32 / 10.0; 256];
        let expected_val = (9 - i) as f32 / 10.0;

        let retrieved = memory.retrieve(&key).unwrap();

        for &val in &retrieved {
            assert!((val - expected_val).abs() < 0.25);
        }
    }
}

#[test]
fn test_interference_resistance() {
    // Test that learning new patterns doesn't catastrophically overwrite old ones
    let mut layer = BTSPLayer::new(100, 2000.0);

    let pattern1 = vec![1.0; 100];
    let pattern2 = vec![0.0, 1.0].iter().cycle().take(100).copied().collect::<Vec<_>>();

    layer.one_shot_associate(&pattern1, 1.0);
    let initial = layer.forward(&pattern1);

    layer.one_shot_associate(&pattern2, 0.5);

    let after_interference = layer.forward(&pattern1);

    // Should retain most of original association
    assert!((initial - after_interference).abs() < 0.3);
}

#[test]
fn test_time_constant_effects() {
    // Short vs long time constants
    let mut short = BTSPLayer::new(50, 500.0);  // 500ms
    let mut long = BTSPLayer::new(50, 5000.0); // 5s

    let pattern = vec![0.5; 50];

    short.one_shot_associate(&pattern, 0.8);
    long.one_shot_associate(&pattern, 0.8);

    // Both should learn equally well
    let short_out = short.forward(&pattern);
    let long_out = long.forward(&pattern);

    assert!((short_out - 0.8).abs() < 0.15);
    assert!((long_out - 0.8).abs() < 0.15);
}

#[test]
fn test_batch_storage_consistency() {
    let mut memory = BTSPAssociativeMemory::new(64, 32);

    let pairs: Vec<(Vec<f32>, Vec<f32>)> = (0..20)
        .map(|i| {
            let key = vec![i as f32 / 20.0; 64];
            let value = vec![(i % 5) as f32 / 5.0; 32];
            (key, value)
        })
        .collect();

    let pair_refs: Vec<_> = pairs.iter().map(|(k, v)| (k.as_slice(), v.as_slice())).collect();
    memory.store_batch(&pair_refs).unwrap();

    // Verify all stored correctly
    for (key, expected_value) in &pairs {
        let retrieved = memory.retrieve(key).unwrap();
        for (exp, act) in expected_value.iter().zip(retrieved.iter()) {
            assert!((exp - act).abs() < 0.3);
        }
    }
}

#[test]
fn test_sparse_pattern_learning() {
    // Test with sparse patterns (realistic for some embeddings)
    let mut layer = BTSPLayer::new(200, 2000.0);

    let mut sparse = vec![0.0; 200];
    for i in (0..200).step_by(20) {
        sparse[i] = 1.0;
    }

    layer.one_shot_associate(&sparse, 0.9);

    let output = layer.forward(&sparse);
    assert!((output - 0.9).abs() < 0.15);
}

#[test]
fn test_scaling_to_large_dimensions() {
    // Test with realistic embedding sizes
    let sizes = vec![384, 768, 1536]; // Common embedding dimensions

    for size in sizes {
        let mut layer = BTSPLayer::new(size, 2000.0);
        let pattern = vec![0.3; size];

        layer.one_shot_associate(&pattern, 0.7);
        let output = layer.forward(&pattern);

        assert!((output - 0.7).abs() < 0.2,
                "Failed at size {}: expected 0.7, got {}", size, output);
    }
}
