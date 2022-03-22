# Optimizing Anchors for Object Detection -- EfficientDet

Process the PascalVOC form dataset into one sole .csv file first using xml_to_csv.py

```python
python xml_to_csv.py --label-dir path/to/labels/directory --save-csv-path path/to/save/csv
```

Then calculate the anchors.

```python
python optimize_anchors.py --path-to-csv path/to/saved/csv --num-clusters 3
```

Then copy the anchors into src/utils/anchors.py
