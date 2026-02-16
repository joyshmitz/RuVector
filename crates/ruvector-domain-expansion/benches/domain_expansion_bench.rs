use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_domain_expansion::{
    ArmId, ContextBucket, CostCurve, CostCurvePoint, ConvergenceThresholds,
    AccelerationScoreboard, DomainExpansionEngine, DomainId, MetaThompsonEngine,
    PolicyKnobs, PopulationSearch, Solution, TransferPrior,
};

fn bench_task_generation(c: &mut Criterion) {
    let engine = DomainExpansionEngine::new();
    let domains = engine.domain_ids();

    let mut group = c.benchmark_group("task_generation");

    for domain_id in &domains {
        group.bench_function(format!("{}", domain_id), |b| {
            b.iter(|| {
                engine.generate_tasks(black_box(domain_id), black_box(10), black_box(0.5))
            })
        });
    }
    group.finish();
}

fn bench_evaluation(c: &mut Criterion) {
    let engine = DomainExpansionEngine::new();
    let rust_id = DomainId("rust_synthesis".into());
    let tasks = engine.generate_tasks(&rust_id, 10, 0.5);

    let solution = Solution {
        task_id: tasks[0].id.clone(),
        content: "fn sum_positives(values: &[i64]) -> i64 { values.iter().filter(|&&x| x > 0).sum() }".into(),
        data: serde_json::Value::Null,
    };

    c.bench_function("evaluate_rust_solution", |b| {
        b.iter(|| {
            let mut eng = DomainExpansionEngine::new();
            eng.evaluate_and_record(
                black_box(&rust_id),
                black_box(&tasks[0]),
                black_box(&solution),
                ContextBucket {
                    difficulty_tier: "medium".into(),
                    category: "transform".into(),
                },
                ArmId("greedy".into()),
            )
        })
    });
}

fn bench_embedding(c: &mut Criterion) {
    let engine = DomainExpansionEngine::new();
    let rust_id = DomainId("rust_synthesis".into());

    let solution = Solution {
        task_id: "bench".into(),
        content: "fn foo() { for i in 0..10 { if i > 5 { let x = i.max(3); } } }".into(),
        data: serde_json::Value::Null,
    };

    c.bench_function("embed_solution", |b| {
        b.iter(|| engine.embed(black_box(&rust_id), black_box(&solution)))
    });
}

fn bench_thompson_sampling(c: &mut Criterion) {
    let mut engine = MetaThompsonEngine::new(vec![
        "greedy".into(),
        "exploratory".into(),
        "conservative".into(),
        "speculative".into(),
    ]);

    let domain = DomainId("bench".into());
    engine.init_domain_uniform(domain.clone());

    let bucket = ContextBucket {
        difficulty_tier: "medium".into(),
        category: "algorithm".into(),
    };

    // Pre-populate with data
    for i in 0..100 {
        let arm = ArmId(format!(
            "{}",
            ["greedy", "exploratory", "conservative", "speculative"][i % 4]
        ));
        let reward = if i % 4 == 0 { 0.9 } else { 0.4 };
        engine.record_outcome(&domain, bucket.clone(), arm, reward, 1.0);
    }

    c.bench_function("thompson_select_arm", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            engine.select_arm(black_box(&domain), black_box(&bucket), &mut rng)
        })
    });
}

fn bench_population_evolve(c: &mut Criterion) {
    let mut search = PopulationSearch::new(16);

    // Pre-populate fitness
    for i in 0..16 {
        if let Some(kernel) = search.kernel_mut(i) {
            kernel.record_score(DomainId("bench".into()), i as f32 / 16.0, 1.0);
        }
    }

    c.bench_function("population_evolve_16", |b| {
        b.iter(|| {
            let mut s = search.clone();
            s.evolve();
        })
    });
}

fn bench_knobs_mutate(c: &mut Criterion) {
    let knobs = PolicyKnobs::default_knobs();
    c.bench_function("knobs_mutate", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            black_box(knobs.mutate(&mut rng, 0.3))
        })
    });
}

fn bench_cost_curve_auc(c: &mut Criterion) {
    let mut curve = CostCurve::new(DomainId("bench".into()), ConvergenceThresholds::default());
    for i in 0..1000 {
        curve.record(CostCurvePoint {
            cycle: i,
            accuracy: (i as f32 / 1000.0).min(1.0),
            cost_per_solve: 1.0 / (i as f32 + 1.0),
            robustness: (i as f32 / 1000.0).min(1.0),
            policy_violations: 0,
            timestamp: i as f64,
        });
    }

    c.bench_function("cost_curve_auc_1000pts", |b| {
        b.iter(|| black_box(curve.auc_accuracy()))
    });
}

fn bench_transfer_prior_extract(c: &mut Criterion) {
    let domain = DomainId("bench".into());
    let mut prior = TransferPrior::uniform(domain);

    // Populate with 100 buckets x 4 arms
    for b in 0..100 {
        for a in 0..4 {
            let bucket = ContextBucket {
                difficulty_tier: format!("tier_{}", b % 3),
                category: format!("cat_{}", b),
            };
            let arm = ArmId(format!("arm_{}", a));
            for _ in 0..20 {
                prior.update_posterior(bucket.clone(), arm.clone(), 0.7);
            }
        }
    }

    c.bench_function("transfer_prior_extract_100buckets", |b| {
        b.iter(|| black_box(prior.extract_summary()))
    });
}

criterion_group!(
    benches,
    bench_task_generation,
    bench_evaluation,
    bench_embedding,
    bench_thompson_sampling,
    bench_population_evolve,
    bench_knobs_mutate,
    bench_cost_curve_auc,
    bench_transfer_prior_extract,
);
criterion_main!(benches);
