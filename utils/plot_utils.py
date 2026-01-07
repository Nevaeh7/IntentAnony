from typing import List, Dict, Tuple

from pathlib import Path
import matplotlib.pyplot as plt
def save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    filename_base: str = "intent_attributes_distribution",
    dpi: int = 300,
    formats: List[str] = None,
) -> Dict[str, Path]:
    """
    Save figure to file (supports PNG and PDF formats)

    Args:
        fig: matplotlib figure object
        output_dir: Output directory
        filename_base: Filename (without extension)
        dpi: Resolution (used for PNG, PDF uses vector format)
        formats: List of save formats, default is ['png', 'pdf']

    Returns:
        Dictionary of saved file paths
    """
    if formats is None:
        formats = ["png", "pdf"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for fmt in formats:
        file_path = output_dir / f"{filename_base}.{fmt}"

        # Use different save parameters for different formats
        if fmt.lower() == "pdf":
            # PDF format: vector format, no DPI, no transparent background
            fig.savefig(
                file_path,
                format="pdf",
                bbox_inches="tight",
                # facecolor="white",
                edgecolor="none",
                transparent=True,
            )
        elif fmt.lower() == "png":
            # PNG format: uses DPI, can use transparent background
            fig.savefig(
                file_path,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                # facecolor="white",
                edgecolor="none",
                transparent=True,
            )
        else:
            # Other formats: use default parameters
            fig.savefig(
                file_path,
                format=fmt,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                transparent=True
            )

        saved_paths[fmt] = file_path
        print(f"✓ {fmt.upper()} saved: {file_path}")

    return saved_paths