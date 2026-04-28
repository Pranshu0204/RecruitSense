# Fine-tuning Dataset

This pipeline uses [`AzharAli05/Resume-Screening-Dataset`](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset) from HuggingFace.

## Auto-download
The dataset is fetched automatically by `prepare_dataset.py` via the `datasets` library — no manual step required.

## Manual download (optional)
```bash
huggingface-cli download AzharAli05/Resume-Screening-Dataset \
  --repo-type dataset \
  --local-dir finetune/dataset/data/
```

## Format
Records are converted to chat-template instruction-tuning format in Phase 8:
- `system`: scoring rubric instructions
- `user`: JD + resume text
- `assistant`: structured JSON score
