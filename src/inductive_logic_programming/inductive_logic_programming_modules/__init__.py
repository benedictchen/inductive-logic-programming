"""
📋   Init  
============

🔬 Research Foundation:
======================
Based on inductive logic programming research:
- Quinlan, J.R. (1990). "Learning Logical Definitions from Relations"
- Muggleton, S. & De Raedt, L. (1994). "Inductive Logic Programming: Theory and Methods"
- Lavrac, N. & Dzeroski, S. (1994). "Inductive Logic Programming: Techniques and Applications"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
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
from .coverage_analysis import (CoverageAnalysisMixin, CoverageMetrics, CoverageAnalysisReport, 
                               calculate_rule_significance as calc_rule_sig, evaluate_coverage_strategy, 
                               generate_coverage_comparison_report)
from .predicate_system import PredicateSystemMixin

__all__ = [
    'LogicalTerm', 'LogicalAtom', 'LogicalClause', 'Example',
    'HypothesisGenerationMixin', 'UnificationEngineMixin', 'SemanticEvaluationMixin',
    'RuleRefinementMixin', 'CoverageAnalysisMixin', 'PredicateSystemMixin',
    'RuleQualityMetrics', 'RefinementStats', 'CoverageMetrics', 'CoverageAnalysisReport',
    'evaluate_semantic_quality', 'compare_semantic_settings',
    'calculate_rule_significance', 'generate_refinement_report',
    'calc_rule_sig', 'evaluate_coverage_strategy', 'generate_coverage_comparison_report'
]

print("""
💰 MODULE SUPPORT - Made possible by Benedict Chen
   ]8;;mailto:benedict@benedictchen.com\benedict@benedictchen.com]8;;\

💰 PLEASE DONATE! Your support keeps this research alive! 💰
   🔗 ]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\💳 CLICK HERE TO DONATE VIA PAYPAL]8;;\
   ❤️ ]8;;https://github.com/sponsors/benedictchen\💖 SPONSOR ON GITHUB]8;;\

   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
   (Start small, dream big! Every donation helps! 😄)
""")
