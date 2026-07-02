TRAIN_CONFIGS tuple: `RANK:ALPHA:DROPOUT:LR:EPOCHS:BATCH:GRAD_ACCUM:CUTOFF:LR_SCHED:WARMUP:SAVE:NUM_GPUS:TARGET`
(effective batch = BATCH × GRAD_ACCUM × NUM_GPUS; TARGET alias: ALL / ATTN / QV / MLP)

Recommended config:
```
256:512:0.0:5.0e-5:1:2:8:16384:cosine:0.05:false:2:ALL
```

Data engine: `min_target_examples = 800`, `episode_filter_top_pct = 100`.
