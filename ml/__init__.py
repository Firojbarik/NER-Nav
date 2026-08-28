"""NER-Nav ML engine package.

Holds the production ML data/feature/training/evaluation/inference logic.
The final model must never be trained on synthetic labels; synthetic data is
used only for unit tests of algorithm mechanics (labelled SYNTHETIC / TEST ONLY).
"""
