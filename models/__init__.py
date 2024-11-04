from .Cheng2020withCKBD import Cheng2020AnchorwithCheckerboard
from .elic import ELIC
from .elic_master import ELIC_master  # luguo
from .elic_united import ELIC_united  # united
from .elic_united_R2D import ELIC_united_R2D  # ICIP chen
from .mlicpp import MLICPlusPlus
from .stf import SymmetricalTransFormer
from .stf_united import SymmetricalTransFormerUnited

# 先找复杂的
modelZoo = {
    "ckbd": Cheng2020AnchorwithCheckerboard,
    "ELIC_united_R2D": ELIC_united_R2D,
    "ELIC_united": ELIC_united,
    "ELIC_master": ELIC_master,
    "ELIC": ELIC,
    "STF_united": SymmetricalTransFormerUnited,
    "STF": SymmetricalTransFormer,
    "MLIC": MLICPlusPlus,
}
