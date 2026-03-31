# TrigramHash + Value Residual + Bank-QAT + GradQuant + MTP + Adam TTT

This run folder implements the experiment approach as a standalone record submission, with the training and output structure aligned to the repository baseline requirements.

## Current Status

- Strategy features are implemented in train_gpt.py.
- Submission metadata has been updated to measured fast-run outputs (not projections).
- Fast local verification run (run_id=local5060_fast_seed1337) reported:
  - final_int6_roundtrip val_bpb: 1.6768
  - final_int6_sliding_window_exact val_bpb: 1.63711028
  - final_int6_sliding_window_s64_exact val_bpb: 1.63503057
  - final submission size (int6+lzma): 5909270 bytes
- Legal score-first TTT was disabled in that fast run configuration (`TTT_ENABLED=0`).

## Strategy Coverage

1. TrigramHash embedding
- Implemented via TrigramHashEmbedding and wired into GPT forward and logits paths.
- Default config uses TRIGRAM_VOCAB_SIZE=4096 and TRIGRAM_DIM=128.

2. Value residual
- Implemented in CausalSelfAttention with vr_lambda blend and v0 carry-over.
- Enabled by default with VALUE_RESIDUAL=1.

3. Bank-level QAT
- Implemented with _fake_quant_int6_ste on bank slices (Q/K/V/O and MLP up/down).
- Controlled by module flag _BANK_QAT_ENABLED.
- Enabled late by threshold and torch.compile recompile path.

4. GradQuant tiered quantization
- Gradient sensitivity accumulation over bank tensors in late training.
- Tier mapping implemented (int5/int6/int7) in mixed_quantize_int6.
- Rebank/unbank export path is implemented and used for quantized roundtrip.

5. Multi-token prediction (MTP)
- MTP heads implemented and trained.
- MTP heads are excluded from export_sd, so no artifact size cost.

6. Adam TTT
- Legal score-first TTT supports Adam mode through TTT_USE_ADAM and TTT_ADAM_LR.

7. Temperature calibration
- Grid search on training tokens only (no validation leakage) before final eval.

8. Extended warmdown and lzma preset 9
- Default WARMDOWN_ITERS=4000.
- Final quant artifact uses lzma.compress(..., preset=9).

## Baseline Compliance Checklist

- Tokenizer-agnostic val_bpb:
  - SentencePiece lookup tables are used to compute byte-aware BPB.
  - Validation loader uses fineweb_val_*.bin shards.

- Data loading structure:
  - load_data_shard validates shard header and expected file size.
  - TokenStream and DistributedTokenLoader follow contiguous token streaming.

- Distributed and time-budget structure:
  - WORLD_SIZE divisibility guard (8 % WORLD_SIZE == 0).
  - Training loop enforces max_wallclock_seconds stopping behavior.

- Artifact and size accounting:
  - Exports final_model.pt and final_model.int6.ptz.
  - Logs model bytes, code bytes, and total submission bytes.

- Final metric/output logging:
  - Logs exact metric lines:
    - final_int6_roundtrip_exact
    - final_int6_sliding_window_exact (when sliding eval is active)
    - final_int6_sliding_window_s64_exact (optional reference)
    - legal_ttt_exact (when TTT is enabled)
  - Logs canonical submission lines for metadata extraction:
    - submission_metric_source
    - submission_metric_exact
    - final_submission_size

## Notes For Record Submission

- Root challenge rules require a reproducible under-10-minute 8xH100 run and statistical significance for SOTA claims.
- The fast local run values here are implementation-validation outputs and are not a full 3-seed SXM significance package.

## Files

- train_gpt.py: Full run implementation
- submission.json: Submission metadata with measured fast-run values
- README.md: Run documentation and compliance notes
- final_model.pt: Exported fp16/bf16 checkpoint artifact
- final_model.int6.ptz: Quantized compressed submission artifact