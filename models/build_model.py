from models.msnet.msnet import MutualSemanticEnhancingNet, MutualSemanticEnhancingNet_lite
from models.braunet.braunet  import BRAUnet
from models.deeplabv3plus.deeplabv3plus import create_deeplabv3plus_resnet50
from models.unet.unet import CustomUNet
from models.unetplusplus.model import CustomUNetPP
from models.joinedtrans.model import UNetPP_doublesmp
from models.rfau.model import ResponseFusionAttentionUConvNextTiny
from models.transunet.model import VisionTransformer
from models.swinunet.model import SwinUnet
from models.eetrans.model import VisionTransformer as EETransUnet
# from models.segformer.segformer import SegFormer

MODEL_ZOO = {
    "msnet": MutualSemanticEnhancingNet,
    "msnetlite":MutualSemanticEnhancingNet_lite,
    # "braunet":BRAUnet,
    # "deeplab":create_deeplabv3plus_resnet50,
    # "unet": CustomUNet,
    # "unetplusplus":CustomUNetPP,
    # "joinedtrans":UNetPP_doublesmp,
    # "rfau":ResponseFusionAttentionUConvNextTiny,
    # "transunet":VisionTransformer,
    # "swinunet":SwinUnet,
    # "eetrans":EETransUnet,
    # "unet": UNet,
    # "segformer": SegFormer,
}
