from .history import HistoryPromptBuilder

import warnings

def create_prompt_builder(config):
    max_history = config.get("max_history", None)
    if max_history is not None:
        warnings.warn("The 'max_history' parameter is deprecated. Please use 'max_text_history' instead.")
    
    max_text_history = max_history
    if max_text_history is None:
        max_text_history = config.max_text_history

    return HistoryPromptBuilder(
        max_text_history=max_text_history,
        max_image_history=config.max_image_history,
        max_cot_history=config.max_cot_history,
    )
