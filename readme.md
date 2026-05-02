# candle-qwen2.5coder-1.5b-minimal

Minimal Rust inference example for **Qwen2.5-Coder 1.5B (quantized)** using Candle.

This project is a stripped-down version of the official Candle example, I recoded it for learning purposes, to get used to the api

## 📦 Model

Designed for:
- **Qwen2.5-Coder 1.5B (quantized)**

> Model weights are downloaded automatically from HuggingFace Hub :)

## 🚀 Usage

### Run the model

```bash
cargo run --release -- \
  --prompt "Write a Rust function for quicksort" \
  --temperature 0.8 \
  --top_p 0.9 \
  --sample_len 200
