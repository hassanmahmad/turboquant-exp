# results/ — experiment outputs

Machine-readable outputs, one directory per track and model tag:

```
results/<track>/<model_tag>/*.json
```

e.g. `results/t1/qwen2.5-7b-instruct/quality_efficiency.json`. Each record should carry the full
config (quantizer, K/V bits, mode, seed, Π regime, `-nc` layers) so a result is self-describing,
and should state whether any memory number is **counted** or **measured**.
