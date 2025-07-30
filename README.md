# Installation

Uhh. IDK create venv, source it, install there some deps like torch, datasets, lightning, and that's it, I believe.

# How to run this AI slop

1. Train tokenizer

```bash
python train_tokenizer.py ./maps/all/ --vocab-limit 2000 --min-frequency 50 --output tiles.json
```

2. Preprocess maps to make it real dataset

```bash
python preprocess.py --map_dir maps/all/ --tokenizer_path tiles.json --output_dir ./dataset --target_size 16 16 --max_layers 16 --augment_variations 10
```

3. Train this scheise

```bash
python train.py --dataset_path ./dataset/ --tokenizer_path ./tiles.json --experiment_name sus1 --batch_size 32 --max_epochs 200 --base_channels 64
```

4. Garbage out

```bash
python inference.py experiments/sus1/checkpoints/last.ckpt --num-maps 1 --temperature 0.6 --tokenizer tiles.json --seed 31337 --output-dir ./generated_slop/
```

ALTERNATIVE ENDING:

```bash
python cont_inference.py experiments/sus1/checkpoints/last.ckpt --num-maps 1 --temperature 0.6 --width 48 --height 64 --tokenizer tiles.json --seed 31337 --output-dir ./long_generated_slop/
```
