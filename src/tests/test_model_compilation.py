from src.models import efficientdet

if __name__ == "__main__":
    model = efficientdet.get_efficientdet(
        name="efficientdet_d0",
        num_classes=20,
        num_anchors=9)
    model.build((None, 512, 512, 3))
    model.summary()