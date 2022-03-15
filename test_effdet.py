from src.models import effdet
from src.models import temp_hparams

if __name__ == "__main__":
    configs = temp_hparams.default_detection_configs()
    model = effdet.EfficientDetNet(
        model_name="efficientdet-d0",
        config=configs)