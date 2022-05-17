# i-Sight: Looking into the Future of Medical AI.

i-Sight incorporates labeled and unlabeled data to create holistic diagnoses of the eye. This attempt focuses primarily on the retinal Fundus and OCT images.

## Demo Video

[![i-Sight Demo]({https://github.com/IdeaKing/i-Sight/blob/main/g3doc/isight_logo.jpg})]({https://youtu.be/09_SoDtCUDg} "i-Sight Demo")

## Sample outputs

Sample fundus output:
![Fundus](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/full_out.png)

Sample OCT output:
![OCT](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/output_oct_od.png)

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
