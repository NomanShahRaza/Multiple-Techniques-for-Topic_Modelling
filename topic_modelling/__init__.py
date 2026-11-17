"""Multiple techniques for topic modelling behind one common interface."""
from .evaluation import coherence, topic_diversity
from .preprocessing import Corpus, load_documents
from .techniques import TECHNIQUES, TopicResult, run_technique

__all__ = ["Corpus", "load_documents", "TECHNIQUES", "TopicResult", "run_technique",
           "coherence", "topic_diversity"]
__version__ = "2.0.0"
