# i-Sight: Looking into the Future of Medical AI.

i-Sight incorporates labeled and unlabeled data to create holistic diagnoses of the eye. This attempt focuses primarily on the retinal Fundus and OCT images.

i-Sight is the winning project of the **Leidos 2022 Young Virginia STEM Research Award**.

## Demo Video

[![i-Sight Demo](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/isight_logo.jpg)](https://youtu.be/09_SoDtCUDg)

## Sample outputs

Sample fundus output:
![Fundus](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/full_out.png)

Sample OCT output:
![OCT](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/output_oct_od.png)

## How it works

i-Sight incorporates labeled and unlabeled data to create holistic diagnoses from multiple points of data. i-Sight uses a novel semi-supervised pseudo learning method called Advanced Meta Pseudo Labels (AMPL) to make diagnosis on a patient's eye health. 

i-Sight solves the most prominent issue in medical AI: the lack of labeled data.
By using the novel semi-supervised training method Advanced Meta Pseudo Labels (AMPL), the i-Sight architecture expands past just retinal images, to CT scans, X-Rays, MRI scans and more. Advanced Meta Pseudo Labels uses 3 neural networks that train each other, allowing for the use of labeled data and unlabeled data.

i-Sight can classify, detect, or segment any medical image when given labeled and unlabeled data. i-Sight can also combine multiple images together to create holistic diagnoses taking into account multiple factors (age, height, ethnicity, etc.) that may play into account on the health of the individual.

![AMPL](https://github.com/IdeaKing/i-Sight/blob/main/g3doc/ampl.png)

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

## Notes

* Backbones and output layers must have a specified dtype of float32.
