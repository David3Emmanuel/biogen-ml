from .image import explain_with_image
from .tabular import explain_with_shap, plot_shap_force, plot_shap_waterfall
from .fused import (
    FusedExplanation,
    explain_prediction,
    explain_both_outputs,
    get_top_features,
    visualize_combined_explanation,
    create_shap_waterfall_plots,
    summarize_prediction
)
