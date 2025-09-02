"""
Inductive Logic Programming Modules
==================================

This package contains modular implementations of ILP components
for learning logical rules from examples and background knowledge.
"""

from .logical_structures import LogicalTerm, LogicalAtom, LogicalClause, Example
from .hypothesis_generation import HypothesisGenerationMixin
from .unification_engine import UnificationEngineMixin
from .semantic_evaluation import SemanticEvaluationMixin, evaluate_semantic_quality, compare_semantic_settings
from .rule_refinement import RuleRefinementMixin, RuleQualityMetrics, RefinementStats, calculate_rule_significance, generate_refinement_report

__all__ = [
    'LogicalTerm', 'LogicalAtom', 'LogicalClause', 'Example',
    'HypothesisGenerationMixin', 'UnificationEngineMixin', 'SemanticEvaluationMixin',
    'RuleRefinementMixin', 'RuleQualityMetrics', 'RefinementStats',
    'evaluate_semantic_quality', 'compare_semantic_settings',
    'calculate_rule_significance', 'generate_refinement_report'
]