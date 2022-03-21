# i-Sight: Looking into the Future of Medical AI.



## Commands for running training

```python
python main.py --train \
               --training-type object_detection \
               --dataset-path datasets/data/VOC2012 \
               --training-dir training_dir/voc \
               --model efficientdet-d0 \
               --debug False \
               --precision float32 \
               --batch-size 8 \
               --epochs 300 \
               --training-method supervised \
               --shuffle-size 16 \
               --image-dims (512, 512) \
               --augment-ds True \
               --print-loss True \
               --log-every-step 100 \
               --max-checkpoints 10 \
               --save-model-frequency 10 \
               --checkpoint-frequency 10
```
