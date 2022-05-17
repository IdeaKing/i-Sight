import tensorflow as tf
from src.models import psp

if __name__ == "__main__":
    model = psp.get_pspnet(name="pspnet_s0")
    model.summary()