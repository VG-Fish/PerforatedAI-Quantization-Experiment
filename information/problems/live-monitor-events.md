
## 2026-09-01 06:16:38 -0400 — fatal worker error


```text
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16826/16838 [1:36:53<00:04,  2.82batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16827/16838 [1:36:53<00:03,  2.89batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16828/16838 [1:36:53<00:03,  2.82batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16829/16838 [1:36:54<00:03,  2.95batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16830/16838 [1:36:54<00:02,  2.90batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16831/16838 [1:36:54<00:02,  2.86batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16832/16838 [1:36:55<00:02,  2.82batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16833/16838 [1:36:55<00:01,  2.83batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16834/16838 [1:36:56<00:01,  2.77batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16835/16838 [1:36:56<00:01,  2.85batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16836/16838 [1:36:56<00:00,  2.80batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|#########9| 16837/16838 [1:36:57<00:00,  2.87batch/s][A
distilbert | dendrites_fp32 | epoch 3/3: 100%|##########| 16838/16838 [1:36:57<00:00,  2.87batch/s][A
                                                                                                   [Adistilbert | dendrites_fp32:  67%|######6   | 2/3 [4:31:23<1:28:26, 5306.47s/epoch, best_accuracy=0.8966, best_epoch=1, val_accuracy=0.8812]distilbert | dendrites_fp32: 100%|##########| 3/3 [4:31:23<00:00, 5544.09s/epoch, best_accuracy=0.8966, best_epoch=1, val_accuracy=0.8812]  distilbert | dendrites_fp32: 100%|##########| 3/3 [4:31:23<00:00, 5427.96s/epoch, best_accuracy=0.8966, best_epoch=1, val_accuracy=0.8812]
[audit] distilbert | dendrites_fp32: no_retained_insertion — raw PAI switch log has no candidate-insertion switch
[06:15:58] [done] distilbert / dendrites_fp32 — Accuracy: 0.9083

Traceback (most recent call last):
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/.venv/bin/dqb", line 12, in <module>
    sys.exit(main())
             ~~~~^^
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/cli.py", line 1163, in main
    _handle_run(args, results_root, comparison_root)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/cli.py", line 901, in _handle_run
    runner.run(
    ~~~~~~~~~~^
        model_keys=selected_models,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        seed=args.seed,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/pipeline.py", line 2074, in run
    newly_trained = self._process_one_model_spec(
        model_spec,
    ...<4 lines>...
        dynamic_dendritic_training,
    )
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/pipeline.py", line 2011, in _process_one_model_spec
    if self._train_pending_condition(
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_spec, condition, bundle,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        model_records, all_records, saved_dirs, allow_pqat,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        dynamic_dendritic_training,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/pipeline.py", line 1904, in _train_pending_condition
    self._require_verified_dendritic_pqat_source(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_spec.key, condition, saved_dirs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/vishy/Desktop/PerforatedAI Quantization Experiment/src/dendritic_benchmark/pipeline.py", line 1824, in _require_verified_dendritic_pqat_source
    raise RuntimeError(
    ...<4 lines>...
    )
RuntimeError: distilbert / dendrites_q8 requires a verified retained dendrites_fp32 source before PQAT; found no_retained_insertion. Run the FP32 dendritic source, inspect its raw PAI switch and architecture evidence, then rerun the PQAT descendants.
```
