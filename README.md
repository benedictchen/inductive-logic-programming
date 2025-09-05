# Inductive Logic Programming

Complete implementation of Inductive Logic Programming algorithms with full research accuracy.
Includes FOIL (Quinlan 1990) and Progol (Muggleton 1995) with comprehensive configuration.

## Features

- **FOIL Algorithm**: Uses information gain to build logical rules step-by-step
- **Progol System**: Uses inverse entailment for hypothesis construction  
- **Predicate System**: Complete vocabulary and type handling
- **Research Accurate**: First-principles implementations with complete theoretical accuracy

## Research Foundation

- Muggleton, S. & De Raedt, L. (1994). "Inductive Logic Programming: Theory and Methods."
- Quinlan, J.R. (1990). "Learning logical definitions from relations."
- Muggleton, S. (1995). "Inverse entailment and Progol."

## Installation

```bash
pip install inductive-logic-programming
```

## Basic Usage

```python
from inductive_logic_programming import FOILLearner, ProgolSystem

# Create a FOIL learner
learner = FOILLearner()

# Learn rules from examples
rules = learner.learn_rules(positive_examples, negative_examples, background_knowledge)
```

## ELI5 Explanation

Inductive Logic Programming is like teaching a computer to discover logical rules from examples.

Imagine showing a computer family tree examples:
- `father(john, mary)` ✓ (john is mary's father)
- `father(bob, alice)` ✓ (bob is alice's father)
- `mother(jane, mary)` ✗ (not a father relationship)

The ILP system learns: "X is father of Y if X is male and parent(X,Y)"

## Author

**Benedict Chen** (benedict@benedictchen.com)

- Support: [PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)
- Sponsor: [GitHub Sponsors](https://github.com/sponsors/benedictchen)

## License

Custom Non-Commercial License with Donation Requirements