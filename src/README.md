Data creation:
1. `src/preprocess.py`
2. `src/make_tensors.py` --> speeds up loading upon training and evaluation

Settings:
1. src/ic_eval4_pretrained.py you can run the model `ydshieh/vit-gpt2-coco-en` as-is and gather evaluation metrics: non-fine-tuned baseline
2. then fine-tune `ydshieh/vit-gpt2-coco-en` and evaluate: fine-tuned baseline
3. train prefix-tuned model and evaluate: more complex approach