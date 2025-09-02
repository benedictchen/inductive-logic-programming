# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/inductive-logic-programming/workflows/CI/badge.svg)](https://github.com/benedictchen/inductive-logic-programming/actions)
[![PyPI version](https://badge.fury.io/py/inductive-logic-programming.svg)](https://badge.fury.io/py/inductive-logic-programming)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Inductive Logic Programming

🧠 FOIL and Progol algorithms for learning logical rules from examples

**Quinlan, J. R. (1990)** - "Learning logical definitions from relations"  
**Muggleton, S. (1995)** - "Inverse entailment and Progol"

## 📦 Installation

```bash
pip install inductive-logic-programming
```

## 🚀 Quick Start

### FOIL Algorithm Example
```python
from inductive_logic_programming import FOIL
import pandas as pd

# Create FOIL learner
foil = FOIL(
    max_variables=5,
    min_positive_coverage=2,
    significance_threshold=0.05
)

# Example: Learning family relationships
# Positive examples: parent(tom, bob), parent(pam, bob), parent(tom, ann)
# Negative examples: parent(bob, tom), parent(ann, pam)

positive_examples = [
    ('parent', ['tom', 'bob']),
    ('parent', ['pam', 'bob']), 
    ('parent', ['tom', 'ann']),
    ('parent', ['bob', 'charlie'])
]

negative_examples = [
    ('parent', ['bob', 'tom']),
    ('parent', ['ann', 'pam']),
    ('parent', ['charlie', 'tom'])
]

# Background knowledge
background = {
    'male': [['tom'], ['bob'], ['charlie']],
    'female': [['pam'], ['ann']],
    'older': [['tom', 'bob'], ['pam', 'bob'], ['tom', 'ann']]
}

# Learn rules
rules = foil.learn(positive_examples, negative_examples, background)
print("Learned rules:", rules)
```

### Progol Algorithm Example  
```python
from inductive_logic_programming import Progol

# Create Progol learner
progol = Progol(
    max_clause_length=5,
    max_search_depth=3,
    compression_required=2
)

# Example: Learning append/3 predicate
examples = {
    'positive': [
        'append([], [1,2], [1,2])',
        'append([1], [2], [1,2])', 
        'append([1,2], [], [1,2])',
        'append([1], [2,3], [1,2,3])'
    ],
    'negative': [
        'append([1], [2], [2,1])',
        'append([1,2], [3], [1,3,2])'
    ]
}

background_knowledge = [
    'list([]).', 
    'list([H|T]) :- list(T).',
    'member(X, [X|_]).',
    'member(X, [_|T]) :- member(X, T).'
]

# Learn clauses
clauses = progol.induce(examples, background_knowledge)
print("Learned clauses:", clauses)
```

## 🔬 Advanced Features

### Rule Refinement
```python
from inductive_logic_programming import RuleRefinement

refiner = RuleRefinement(
    refinement_operator='rho',
    completeness_check=True,
    consistency_check=True
)

# Refine an initial hypothesis
initial_rule = "parent(X, Y) :- older(X, Y)"
refined_rules = refiner.refine(
    initial_rule, 
    positive_examples, 
    negative_examples,
    background
)
```

### Custom Predicate Learning
```python
from inductive_logic_programming import PredicateLearner

# Learn custom predicates with domain-specific knowledge
learner = PredicateLearner(
    target_predicate='grandparent',
    mode_declarations=[
        'grandparent(+person, +person)',
        'parent(+person, -person)',
        'parent(-person, +person)'
    ]
)

examples = [
    'grandparent(tom, charlie)',
    'grandparent(pam, charlie)'
]

learned_def = learner.induce_definition(examples, background)
```

## 🧬 Key Algorithmic Features

### FOIL Algorithm
- **Information Gain Heuristic**: Selects literals that maximize information gain
- **Pruning Strategies**: Eliminates unpromising search paths early
- **Significance Testing**: Statistical validation of learned rules
- **Incremental Learning**: Can learn from streaming examples

### Progol System  
- **Mode-Directed Inverse Entailment**: Efficient bottom-up clause construction
- **Compression-Based Learning**: Prioritizes hypotheses with high compression
- **Clause Refinement**: Systematic search through hypothesis space
- **Background Knowledge Integration**: Seamless use of domain knowledge

### Rule Quality Metrics
- **Coverage**: Number of positive examples explained by rule
- **Precision**: Ratio of correctly classified positive examples  
- **Compression**: Reduction in description length
- **Statistical Significance**: Confidence in learned patterns

## 📊 Implementation Highlights

- **Research Accuracy**: Faithful implementation of original algorithms
- **Logic Programming Integration**: Full Prolog compatibility
- **Scalable Learning**: Handles large datasets efficiently
- **Educational Value**: Clear implementation for learning ILP concepts
- **Extensible Framework**: Easy to add new learning algorithms

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**