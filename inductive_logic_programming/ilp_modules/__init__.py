"""
Inductive Logic Programming Modular Components

Based on: Muggleton & De Raedt (1994) "Inductive Logic Programming: Theory and Methods"
"""

from .logical_structures import LogicalTerm, LogicalAtom, LogicalClause, Example
from .inductive_logic_programmer import InductiveLogicProgrammer

__all__ = ['LogicalTerm', 'LogicalAtom', 'LogicalClause', 'Example', 'InductiveLogicProgrammer']
