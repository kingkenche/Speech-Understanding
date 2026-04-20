"""Part III: speaker cloning and prosody transfer modules."""

from .dtw_warping import dtw_align, warp_contour
from .prosody_extractor import ProsodyFeatures, extract_prosody
from .speaker_embedding import extract_speaker_embedding
from .synthesizer import apply_prosody_warping, synthesize_text

__all__ = [
	"ProsodyFeatures",
	"apply_prosody_warping",
	"dtw_align",
	"extract_prosody",
	"extract_speaker_embedding",
	"synthesize_text",
	"warp_contour",
]
