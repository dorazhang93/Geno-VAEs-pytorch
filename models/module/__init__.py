from .encoder import *
from .decoder import *

Encoders = {"EncoderConv1D":ConvEncoder,
            'EncoderTransformer':Enformer,
            'EncoderBasenjiDiConv':EnBasenjiDiConv,
            'EncoderBasenjiMLP':EnBasenjiMlp,
            }

Decoders = {"DecoderConv1D":ConvDecoder,
            'DecoderTransformer':Deformer,
            'DecoderBasenjiDiConv':DeBasenjiDiConv,
            'DecoderBasenjiMLP':DeBasenjiMlp,
            }