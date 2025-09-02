"""
Inductive Logic Programming Library
Based on: Muggleton & De Raedt (1994) "Inductive Logic Programming: Theory and Methods"

This library implements learning of logical rules from examples and background knowledge,
combining symbolic reasoning with machine learning for interpretable rule discovery.

Core Research Concepts Implemented:
• Rule Learning: Automated discovery of logical rules from examples
• Logic Programs: First-order logic representation of learned knowledge  
• FOIL Algorithm: Top-down rule induction with information gain heuristics
• Progol System: Bottom-up learning using inverse entailment and compression
• Hypothesis Refinement: Systematic improvement of learned hypotheses through specialization and generalization
"""

def _print_attribution():
    """Print attribution message with donation link"""
    try:
        print("\n🧠 Inductive Logic Programming Library - Made possible by Benedict Chen")
        print("   \033]8;;mailto:benedict@benedictchen.com\033\\benedict@benedictchen.com\033]8;;\033\\")
        print("   Support his work: \033]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\033\\🍺 Buy him a beer\033]8;;\033\\")
    except:
        print("\n🧠 Inductive Logic Programming Library - Made possible by Benedict Chen")
        print("   benedict@benedictchen.com")
        print("   Support: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS")

from .inductive_logic_programming import (
    InductiveLogicProgrammer,
    LogicalTerm,
    LogicalAtom,
    LogicalClause,
    Example
)
from .foil import FOILLearner, FOILStatistics
from .progol import ProgolSystem, ProgolSettings, ProgolStatistics
from .rule_refinement import RuleRefinement, RefinementOperator, SpecializationOperator, GeneralizationOperator

# Show attribution on library import
_print_attribution()

# Add ILPConfig for test compatibility
from dataclasses import dataclass
from typing import Optional

@dataclass
class ILPConfig:
    """Configuration class for ILP systems"""
    algorithm: str = 'foil'
    max_clauses: int = 100
    min_accuracy: float = 0.8
    noise_level: float = 0.0
    max_clause_length: int = 5
    max_variables: int = 4

# Create wrapper class for test compatibility
class InductiveLogicProgramming:
    """Wrapper for InductiveLogicProgrammer with test-compatible interface"""
    
    def __init__(self, algorithm: str = 'foil', max_clauses: int = 100, 
                 config: Optional[ILPConfig] = None, **kwargs):
        """Initialize ILP system with test-compatible parameters"""
        
        if config is not None:
            self.algorithm = config.algorithm
            self.max_clauses = config.max_clauses
            self.min_accuracy = config.min_accuracy
            self.noise_level = config.noise_level
            max_clause_length = config.max_clause_length
            max_variables = config.max_variables
        else:
            self.algorithm = algorithm
            self.max_clauses = max_clauses
            self.min_accuracy = kwargs.get('min_accuracy', 0.8)
            self.noise_level = kwargs.get('noise_level', 0.0)
            max_clause_length = kwargs.get('max_clause_length', 5)
            max_variables = kwargs.get('max_variables', 4)
        
        # Create underlying ILP system
        self._ilp = InductiveLogicProgrammer(
            max_clause_length=max_clause_length,
            max_variables=max_variables,
            confidence_threshold=self.min_accuracy,
            noise_tolerance=self.noise_level,
            **kwargs
        )
    
    def __getattr__(self, name):
        """Delegate all method calls to the wrapped instance"""
        return getattr(self._ilp, name)

# Create aliases for test compatibility
FOILAlgorithm = FOILLearner
ProgolAlgorithm = ProgolSystem

__version__ = "1.0.0"
__authors__ = ["Based on Muggleton & De Raedt (1994)"]

__all__ = [
    "InductiveLogicProgrammer",
    "InductiveLogicProgramming",  # New wrapper class
    "ILPConfig",  # New config class
    "LogicalTerm",
    "LogicalAtom", 
    "LogicalClause",
    "Example",
    "FOILLearner",
    "FOILAlgorithm",  # Alias
    "FOILStatistics",
    "ProgolSystem",
    "ProgolAlgorithm",  # Alias
    "ProgolSettings", 
    "ProgolStatistics",
    "RuleRefinement",
    "RefinementOperator",
    "SpecializationOperator",
    "GeneralizationOperator"
]