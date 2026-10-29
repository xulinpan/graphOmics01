"""tcsimpute: transcript-conditioned selection imputation of left-censored proteomics.

A multimodal variational autoencoder whose protein likelihood is a probit selection model, so that
values missing below a detection limit are imputed from a paired transcriptome.

Quick start::

    from tcsimpute import TCSImputer
    imp = TCSImputer(limit_scale="raw").fit(rna_df, protein_df)   # NaN = missing protein
    imputed = imp.transform(rna_df, protein_df)
    lower, upper = imp.predict_interval(rna_df, protein_df)
"""

from .imputer import TCSImputer, match_cognates
from .model import CensoredMultimodalVAE, VAEConfig, train_model

__version__ = "0.2.3"
__all__ = ["TCSImputer", "match_cognates", "CensoredMultimodalVAE", "VAEConfig", "train_model",
           "__version__"]
