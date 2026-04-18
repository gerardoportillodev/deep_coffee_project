#!/usr/bin/env python3
"""Quick test to verify all Stage 1 modules can be imported."""

import sys

modules = [
    ("configs.config", ["Config", "get_config"]),
    ("src.data.dataset", ["CoffeeBeansDataset", "collect_samples", "get_dataloaders"]),
    ("src.data.eda", ["class_distribution", "run_stage1_eda"]),
    ("src.data.transforms", ["get_train_transforms", "get_eval_transforms"]),
    ("src.data.split", ["split_train_val_samples"]),
    ("src.models.mlp", ["CoffeeMLP"]),
    ("src.training.trainer", ["Trainer"]),
    ("src.training.engine", ["run_inference"]),
    ("src.evaluation.metrics", ["compute_classification_metrics"]),
    ("src.utils.seed", ["set_seed"]),
]

all_ok = True
for module_name, exports in modules:
    try:
        mod = __import__(module_name, fromlist=exports)
        for export in exports:
            if not hasattr(mod, export):
                print(f"✗ {module_name}.{export} NOT FOUND")
                all_ok = False
        if all_ok:
            print(f"✓ {module_name}")
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        all_ok = False

if all_ok:
    print("\n✅ All Stage 1 modules are functional!")
    sys.exit(0)
else:
    print("\n❌ Some modules have issues")
    sys.exit(1)
