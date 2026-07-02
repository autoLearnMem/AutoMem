TRAIN_CONFIGS tuple: `RANK:ALPHA:DROPOUT:LR:EPOCHS:BATCH:GRAD_ACCUM:CUTOFF:LR_SCHED:WARMUP:SAVE:NUM_GPUS:TARGET`
(effective batch = BATCH × GRAD_ACCUM × NUM_GPUS; TARGET alias: ALL / ATTN / QV / MLP)

Recommended config:
```
128:256:0.0:5.0e-5:3:2:4:16384:cosine:0.05:false:2:ALL
```

Data engine: `min_target_examples = 300`, `episode_filter_top_pct = 100`.
