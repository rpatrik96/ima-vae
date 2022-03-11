#!/bin/bash
python3 ima_vae/cli.py fit --config configs/trainer.yaml --config configs/synth/moebius.yaml --trainer.profiler=simple --trainer.max_epochs=2