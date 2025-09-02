"""
🧠 Inductive Logic Programming - Learning Logical Rules from Examples
====================================================================

Author: Benedict Chen (benedict@benedictchen.com)

💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to fully support continued research

Based on: Muggleton & De Raedt (1994) "Inductive Logic Programming: Theory and Methods"

🎯 ELI5 Summary:
Think of this as a super-smart detective that learns rules by looking at examples! 
You show it examples like "John is Mary's father" and "Bob is Alice's father", plus 
some background knowledge about families, and it figures out the rule: "If X is Y's 
father, then X is a parent of Y". It's like teaching a computer to be Sherlock Holmes!

🔬 Research Background:
========================
Stephen Muggleton and Luc De Raedt's 1994 breakthrough created the field of 
Inductive Logic Programming (ILP). This solved a fundamental AI challenge: 
how to automatically learn interpretable logical rules from data.

The ILP revolution:
- Combines symbolic reasoning with statistical learning
- Learns human-readable rules (not black boxes)
- Uses background knowledge to guide learning
- Handles noisy and incomplete data
- Enables explainable AI before it was trendy

This launched the field of "relational learning" and influenced modern 
approaches like neural-symbolic integration and graph neural networks.

🏗️ Architecture:
================
Examples + Background Knowledge → Hypothesis Generation → Rule Refinement → Learned Rules

🎨 ASCII Diagram - ILP Learning Process:
======================================
Background Knowledge     Examples (+/-)        Learning Process
     ┌─────────────┐     ┌─────────────┐      ┌─────────────────┐
     │ parent(X,Y) │     │ +father(j,m)│ ──→  │ 1. Generate     │
     │ male(X)     │     │ +father(b,a)│      │    Hypotheses   │
     │ female(X)   │     │ -father(m,j)│ ──→  │ 2. Test Coverage│
     │ ...         │     │ ...         │      │ 3. Refine Rules │
     └─────────────┘     └─────────────┘      │ 4. Select Best  │
            ↓                    ↓             └─────────────────┘
            └────────────────────┴─────────────────────↓
                                              ┌─────────────────┐
                                              │ Learned Rules:  │
                                              │ father(X,Y) :-  │
                                              │   parent(X,Y),  │
                                              │   male(X)       │
                                              └─────────────────┘

Mathematical Framework:
- Hypothesis: H (set of logical clauses)
- Background Knowledge: B (known facts and rules)
- Examples: E+ (positive) and E- (negative)
- Goal: Find H such that B ∧ H ⊨ E+ and B ∧ H ∧ E- ⊭ ⊥

🚀 Key Innovation: Interpretable Rule Learning
Revolutionary Impact: Automated discovery of symbolic knowledge from data

⚡ Learning Methods:
===================
✨ Semantic Settings:
  - Normal: Classical logic semantics with consistency
  - Definite: Definite clause semantics (Horn clauses)
  - Nonmonotonic: Closed-world assumption with minimality

✨ Search Strategies:
  - Top-down: Start general, specialize (like FOIL)
  - Bottom-up: Start specific, generalize (like Progol)
  - Hybrid: Combine both approaches

✨ Rule Refinement:
  - Specialization: Add conditions to reduce overgeneralization
  - Generalization: Remove conditions to increase coverage
  - Predicate invention: Create new intermediate concepts

✨ Advanced Features:
  - Noise tolerance: Handle incorrect/inconsistent examples
  - Predicate hierarchies: Use type information for better rules
  - Multi-predicate learning: Learn sets of interrelated rules
  - Statistical significance: Ensure learned rules are meaningful

Key Innovation: Bridging the gap between symbolic AI and machine learning,
enabling automated discovery of interpretable knowledge from relational data!
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Set
from dataclasses import dataclass
from itertools import product, combinations
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LogicalTerm:
    """Represents a logical term (constant, variable, or function)"""
    name: str
    term_type: str  # 'constant', 'variable', 'function'
    arguments: Optional[List['LogicalTerm']] = None
    
    def __str__(self):
        if self.term_type == 'function' and self.arguments:
            args_str = ", ".join(str(arg) for arg in self.arguments)
            return f"{self.name}({args_str})"
        return self.name


@dataclass  
class LogicalAtom:
    """Represents a logical atom (predicate with terms)"""
    predicate: str
    terms: List[LogicalTerm]
    negated: bool = False
    
    def __str__(self):
        terms_str = ", ".join(str(term) for term in self.terms)
        atom_str = f"{self.predicate}({terms_str})"
        return f"¬{atom_str}" if self.negated else atom_str


@dataclass
class LogicalClause:
    """Represents a logical clause (Horn clause: head :- body)"""
    head: LogicalAtom
    body: List[LogicalAtom]
    confidence: float = 1.0
    
    def __str__(self):
        if not self.body:
            return str(self.head)
        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}"


@dataclass
class Example:
    """Training example (positive or negative)"""
    atom: LogicalAtom
    is_positive: bool
    
    def __str__(self):
        sign = "+" if self.is_positive else "-"
        return f"{sign} {self.atom}"


