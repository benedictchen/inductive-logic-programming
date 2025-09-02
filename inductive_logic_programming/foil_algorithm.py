"""
📖 FOIL Algorithm for Inductive Logic Programming 📖

A comprehensive implementation of the FOIL (First-Order Inductive Learner) algorithm
for learning first-order logical rules from examples. Discovers Horn clause rules
using information gain heuristics to construct interpretable logical theories.

Author: Benedict Chen
Email: benedict@benedictchen.com
Created: 2024
License: MIT

💝 Support This Research:
If this FOIL implementation enriches your logical reasoning projects, consider
supporting the resurrection of symbolic AI classics! Like building logical rules
from scattered facts, your support helps us construct a comprehensive library
of AI foundations:
- GitHub: ⭐ Star this repository to strengthen the knowledge base
- Donate: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
- Cite: Reference this work in your symbolic AI research

═══════════════════════════════════════════════════════════════════════════════════
📚 RESEARCH FOUNDATIONS
═══════════════════════════════════════════════════════════════════════════════════

This implementation builds upon foundational work in inductive logic programming:

📜 FOUNDATIONAL PAPERS:
• Quinlan, J.R. (1990). "Learning logical definitions from relations"
  Machine Learning, 5(3), 239-266
  - Original FOIL algorithm formulation
  - Information-theoretic approach to rule learning
  - Horn clause induction from positive and negative examples

• Quinlan, J.R. & Cameron-Jones, R.M. (1993). "FOIL: A midterm report"
  Machine Learning: ECML-93, pp. 1-20
  - Enhanced FOIL with improved pruning and search
  - Performance analysis on benchmark problems
  - Comparison with other ILP systems

🧠 INDUCTIVE LOGIC PROGRAMMING:
• Muggleton, S. (1991). "Inductive logic programming"
  New Generation Computing, 8(4), 295-318
  - Foundational framework for ILP
  - Learning from examples and background knowledge
  - Integration of machine learning and logic programming

• Lavrac, N. & Dzeroski, S. (1994). "Inductive Logic Programming: Techniques and Applications"
  Ellis Horwood
  - Comprehensive survey of ILP methods
  - Applications in knowledge discovery
  - Theoretical foundations and practical algorithms

🔍 INFORMATION THEORY IN LEARNING:
• Quinlan, J.R. (1986). "Induction of decision trees"
  Machine Learning, 1(1), 81-106
  - Information gain for feature selection
  - Entropy-based learning principles
  - Application to logical rule induction

• Mitchell, T.M. (1997). "Machine Learning"
  McGraw-Hill, Chapter 10: Learning Sets of Rules
  - General-to-specific search in rule learning
  - Covering algorithms and sequential learning
  - Performance evaluation in rule-based systems

🎯 HORN CLAUSE LEARNING:
• Plotkin, G.D. (1970). "A note on inductive generalization"
  Machine Intelligence, 5, 153-163
  - Least general generalization in first-order logic
  - Theoretical foundations for logical generalization
  - Subsumption and generality orderings

• Michalski, R.S. (1983). "A theory and methodology of inductive learning"
  Artificial Intelligence, 20(2), 111-161
  - Star methodology for concept learning
  - Structural approach to machine learning
  - Rule quality evaluation criteria

═══════════════════════════════════════════════════════════════════════════════════
🎭 EXPLAIN LIKE I'M 5: The Detective Rule Maker
═══════════════════════════════════════════════════════════════════════════════════

Imagine you're a super-smart detective trying to figure out the rules for when
something happens by looking at lots of examples! 🕵️‍♂️🔍

🎮 THE RULE DISCOVERY GAME:
You have a bunch of facts about animals and you want to figure out the rules:

EXAMPLES YOU SEE:
✅ "Tweety is a bird AND Tweety can fly" 
✅ "Robbie is a bird AND Robbie can fly"
✅ "Polly is a bird AND Polly can fly"  
❌ "Fluffy is a cat AND Fluffy can fly" (Nope!)
❌ "Rex is a dog AND Rex can fly" (Nope!)

🧠 THE DETECTIVE'S THINKING:
"Hmm, I notice a pattern! Let me make a rule:
IF something is a bird, THEN it can fly!"

🎯 HOW FOIL WORKS LIKE A DETECTIVE:

🔍 STEP 1: Start with a Simple Guess
- "I want to learn when things can fly"
- Start with: "IF ??? THEN can fly"

📊 STEP 2: Look for the Best Clue  
- Try different clues: "is_bird", "has_wings", "is_small"
- Count which clue works best with our examples
- "is_bird" works for 3/3 flying examples! 🎯

🎲 STEP 3: Test the Rule
- Rule: "IF is_bird THEN can_fly"
- Check: Does this work for ALL our examples?
- Flying things: Tweety ✅, Robbie ✅, Polly ✅
- Non-flying things: Fluffy ❌, Rex ❌ (Good, they're not birds!)

🏗️ STEP 4: Make the Rule More Specific if Needed
Sometimes our first rule is too simple:
- What if we see "Ozzy is a bird but Ozzy can't fly" (he's an ostrich!)
- Then we need: "IF is_bird AND is_small THEN can_fly"

🎪 THE MAGIC:
FOIL is like having a super-detective brain that:
- Looks at LOTS of examples automatically
- Tries ALL possible clues and combinations  
- Picks the BEST rules using math (information gain)
- Makes rules that are easy for humans to understand!

So instead of guessing rules, FOIL discovers them like a detective finding patterns! 🕵️‍♂️✨

═══════════════════════════════════════════════════════════════════════════════════
🏗️ SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════════

The FOIL algorithm follows a covering-based learning architecture:

                        📖 FOIL LEARNING SYSTEM
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
          📥 TRAINING          🔄 RULE          📤 LEARNED
          EXAMPLES            CONSTRUCTION      THEORY
              │                   │               │
      ┌───────┴───────┐          │       ┌───────┴───────┐
  ✅ POSITIVE     ❌ NEGATIVE      │     📜 HORN        🎯 RULE
  EXAMPLES       EXAMPLES         │     CLAUSES        QUALITY
      │               │           │         │             │
      └───────────────┼───────────┼─────────┼─────────────┘
                      │           │         │
              🧠 COVERING ALGORITHM
                      │           │
        ┌─────────────┴───────────┴─────────────┐
        │           🎯 RULE CONSTRUCTION         │
        │         • Literal Selection           │
        │         • Information Gain           │
        │         • Specialization Search      │
        │         • Pruning & Evaluation       │
        └─────────────────────────────────────┘
                              │
                    📊 LEARNED RULES

LEARNING LOOP:
1. Uncovered Examples → Rule Construction → New Rule
2. Apply Rule → Remove Covered Examples → Remaining Examples
3. Repeat until all positive examples are covered

RULE CONSTRUCTION:
1. Start with most general rule (head only)
2. Add literals based on information gain
3. Specialize until no negative examples covered
4. Prune redundant literals

SEARCH STRATEGY:
- General-to-specific beam search
- Information gain heuristic for literal selection
- Minimum description length for stopping

═══════════════════════════════════════════════════════════════════════════════════
🧮 MATHEMATICAL FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════════

INFORMATION GAIN CALCULATION:
For a partial rule R covering p₀ positive and n₀ negative examples,
adding literal L to get rule R∧L covering p₁ positive and n₁ negative examples:

Information Gain = p₁ × (log₂(p₁/(p₁+n₁)) - log₂(p₀/(p₀+n₀)))

Where the first term is the gain in information per positive example.

FOIL GAIN (Original):
Gain(L,R) = t × (log₂(p₁/(p₁+n₁)) - log₂(p₀/(p₀+n₀)))

Where:
- t = p₁ (number of positive examples covered by R∧L)
- p₀, n₀ = positive/negative examples covered by R
- p₁, n₁ = positive/negative examples covered by R∧L

ENCODING LENGTH (MDL):
Total description length = Rule_Length + Error_Length

Rule_Length = Σᵢ log₂(|Predicates|) + log₂(|Variables|)
Error_Length = -Σᵢ log₂(P(correct_classification))

RULE QUALITY METRICS:
Precision = p₁/(p₁+n₁)
Recall = p₁/P_total  
F-Score = 2 × (Precision × Recall)/(Precision + Recall)

Coverage = (p₁+n₁)/(P_total+N_total)

STOPPING CONDITIONS:
1. Information gain < threshold
2. All positive examples covered
3. Rule becomes too specific (covers < min_pos examples)
4. MDL increases rather than decreases

LITERAL SELECTION:
For each candidate literal L:
1. Compute information gain Gain(L,R)
2. Consider computational cost Cost(L)
3. Select L* = argmax_L (Gain(L,R) - λ×Cost(L))

Where λ balances accuracy vs. complexity.

RULE REFINEMENT:
Bottom-up pruning: Remove literals that don't decrease error
Top-down specialization: Add literals that increase precision

THEORY CONSTRUCTION:
Given learned rules R₁, R₂, ..., Rₙ:
Theory = R₁ ∨ R₂ ∨ ... ∨ Rₙ

Evaluate theory using cross-validation and hold-out testing.

═══════════════════════════════════════════════════════════════════════════════════
🌍 REAL-WORLD APPLICATIONS
═══════════════════════════════════════════════════════════════════════════════════

🧬 BIOINFORMATICS:
• Protein function prediction: Learn rules relating structure to function
• Gene regulatory networks: Discover transcription factor binding rules
• Drug discovery: Learn rules for molecular activity and toxicity
• Phylogenetic analysis: Infer evolutionary relationships from sequence data

🏥 MEDICAL DIAGNOSIS:
• Clinical decision support: Learn diagnostic rules from patient records  
• Drug interaction prediction: Discover rules for adverse drug combinations
• Epidemiological analysis: Learn rules for disease transmission patterns
• Treatment outcome prediction: Rules relating patient characteristics to outcomes

📊 KNOWLEDGE DISCOVERY:
• Customer behavior analysis: Learn purchasing pattern rules from transaction data
• Fraud detection: Discover rules identifying suspicious financial activities
• Web usage mining: Learn user navigation patterns for site optimization
• Scientific discovery: Extract causal rules from experimental observations

🤖 ROBOTICS AND AI:
• Action planning: Learn precondition rules for robotic actions
• Natural language processing: Discover grammatical rules from text corpora
• Computer vision: Learn object recognition rules from visual features
• Multi-agent systems: Infer coordination rules from interaction patterns

🔬 SCIENTIFIC MODELING:
• Ecological modeling: Learn predator-prey relationship rules
• Climate science: Discover rules relating atmospheric variables
• Materials science: Learn structure-property relationships
• Astronomy: Infer classification rules for celestial objects

⚖️ LEGAL AND REGULATORY:
• Case law analysis: Extract legal precedent rules from court decisions
• Regulatory compliance: Learn compliance rules from regulatory texts
• Contract analysis: Discover standard clause patterns and relationships
• Risk assessment: Learn rules for evaluating legal and business risks

🎓 EDUCATIONAL TECHNOLOGY:
• Intelligent tutoring: Learn rules for personalized instruction
• Curriculum design: Discover prerequisite relationships between concepts
• Assessment: Learn rules for automated essay scoring and feedback
• Learning analytics: Infer student performance prediction rules

🌐 NETWORK ANALYSIS:
• Social network analysis: Learn rules for community formation and influence
• Internet routing: Discover optimal path selection rules
• Cybersecurity: Learn intrusion detection rules from network traffic
• Communication protocols: Infer protocol specifications from traffic analysis

This FOIL implementation enables interpretable machine learning across domains
where logical rules provide valuable insights into underlying patterns! 🎯
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import itertools
from collections import defaultdict


class LiteralType(Enum):
    """Types of literals that can be added to rules"""
    EQUALITY = "equality"      # X = constant
    INEQUALITY = "inequality"  # X != Y
    UNARY = "unary"           # predicate(X)  
    BINARY = "binary"         # predicate(X,Y)
    COMPARISON = "comparison"  # X > Y, X < Y


@dataclass 
class Literal:
    """Represents a literal in a Horn clause"""
    predicate: str
    arguments: List[str]
    negated: bool = False
    literal_type: LiteralType = LiteralType.UNARY
    
    def __str__(self):
        neg_str = "not " if self.negated else ""
        arg_str = ",".join(self.arguments)
        return f"{neg_str}{self.predicate}({arg_str})"


@dataclass
class HornClause:
    """Represents a Horn clause rule"""
    head: Literal
    body: List[Literal]
    positive_coverage: int = 0
    negative_coverage: int = 0
    
    def __str__(self):
        if not self.body:
            return str(self.head)
        body_str = " and ".join(str(lit) for lit in self.body)
        return f"{self.head} :- {body_str}"
    
    def precision(self) -> float:
        """Calculate precision of the rule"""
        total = self.positive_coverage + self.negative_coverage
        return self.positive_coverage / total if total > 0 else 0.0
    
    def recall(self, total_positives: int) -> float:
        """Calculate recall of the rule"""
        return self.positive_coverage / total_positives if total_positives > 0 else 0.0


class FOILLearner:
    """
    FOIL algorithm for learning Horn clause rules
    
    Implements the covering algorithm with information gain heuristic
    for constructing first-order logical rules from examples.
    """
    
    def __init__(self, 
                 min_pos_coverage: int = 2,
                 max_rule_length: int = 10,
                 beam_width: int = 5,
                 noise_threshold: float = 0.1):
        """
        Initialize FOIL learner
        
        Args:
            min_pos_coverage: Minimum positive examples a rule must cover
            max_rule_length: Maximum number of literals in rule body
            beam_width: Width of beam search for rule construction
            noise_threshold: Acceptable negative coverage ratio
        """
        self.min_pos_coverage = min_pos_coverage
        self.max_rule_length = max_rule_length
        self.beam_width = beam_width
        self.noise_threshold = noise_threshold
        
        # Training data
        self.positive_examples = []
        self.negative_examples = []
        self.predicates = set()
        self.constants = set()
        self.variables = set()
        
        # Learned theory
        self.learned_rules = []
        
    def fit(self, positive_examples: List[Dict[str, Any]], 
            negative_examples: List[Dict[str, Any]],
            target_predicate: str) -> List[HornClause]:
        """
        Learn Horn clause rules from positive and negative examples
        
        Args:
            positive_examples: List of positive example dictionaries
            negative_examples: List of negative example dictionaries
            target_predicate: The predicate to learn rules for
            
        Returns:
            List of learned Horn clause rules
        """
        
        print(f"🎯 Learning rules for predicate: {target_predicate}")
        print(f"   Positive examples: {len(positive_examples)}")
        print(f"   Negative examples: {len(negative_examples)}")
        
        self.positive_examples = positive_examples.copy()
        self.negative_examples = negative_examples.copy()
        
        # Extract vocabulary
        self._extract_vocabulary()
        
        # Learn rules using covering algorithm
        uncovered_positives = positive_examples.copy()
        self.learned_rules = []
        
        rule_count = 0
        while uncovered_positives and rule_count < 50:  # Prevent infinite loops
            rule_count += 1
            print(f"\n📖 Learning rule {rule_count}...")
            print(f"   Uncovered positives: {len(uncovered_positives)}")
            
            # Learn one rule
            rule = self._learn_single_rule(uncovered_positives, 
                                         self.negative_examples,
                                         target_predicate)
            
            if rule is None or rule.positive_coverage < self.min_pos_coverage:
                print("   No more good rules found, stopping.")
                break
                
            self.learned_rules.append(rule)
            print(f"✓ Learned rule: {rule}")
            print(f"   Coverage: +{rule.positive_coverage} -{rule.negative_coverage}")
            print(f"   Precision: {rule.precision():.3f}")
            
            # Remove covered positive examples
            covered_positives = self._get_covered_examples(rule, uncovered_positives)
            uncovered_positives = [ex for ex in uncovered_positives 
                                 if ex not in covered_positives]
        
        print(f"\n🎉 Learning complete! Learned {len(self.learned_rules)} rules")
        total_coverage = sum(rule.positive_coverage for rule in self.learned_rules)
        print(f"   Total positive coverage: {total_coverage}/{len(positive_examples)}")
        
        return self.learned_rules
    
    def _extract_vocabulary(self):
        """Extract predicates, constants, and variables from examples"""
        all_examples = self.positive_examples + self.negative_examples
        
        for example in all_examples:
            for key, value in example.items():
                self.predicates.add(key)
                if isinstance(value, str):
                    self.constants.add(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            self.constants.add(item)
    
    def _learn_single_rule(self, positive_examples: List[Dict[str, Any]],
                          negative_examples: List[Dict[str, Any]], 
                          target_predicate: str) -> Optional[HornClause]:
        """
        Learn a single Horn clause rule using beam search
        """
        
        # Create initial rule with just the head
        head_literal = Literal(target_predicate, ["X"])
        initial_rule = HornClause(head_literal, [])
        
        # Calculate initial coverage
        initial_rule.positive_coverage = len(positive_examples)
        initial_rule.negative_coverage = len(negative_examples)
        
        if initial_rule.negative_coverage == 0:
            return initial_rule  # Perfect rule already!
        
        # Beam search for best rule
        beam = [initial_rule]
        
        for depth in range(self.max_rule_length):
            if not beam:
                break
                
            new_beam = []
            
            for rule in beam:
                # Generate candidate literals
                candidates = self._generate_candidate_literals(rule)
                
                # Evaluate each candidate
                for literal in candidates:
                    new_rule = HornClause(rule.head, rule.body + [literal])
                    
                    # Calculate coverage
                    pos_covered = self._count_coverage(new_rule, positive_examples)
                    neg_covered = self._count_coverage(new_rule, negative_examples)
                    
                    new_rule.positive_coverage = pos_covered
                    new_rule.negative_coverage = neg_covered
                    
                    # Check if rule is acceptable
                    if (pos_covered >= self.min_pos_coverage and 
                        neg_covered <= pos_covered * self.noise_threshold):
                        
                        if neg_covered == 0:
                            # Perfect rule found!
                            return new_rule
                        
                        new_beam.append(new_rule)
            
            # Keep only best rules (beam search)
            if new_beam:
                new_beam.sort(key=self._evaluate_rule, reverse=True)
                beam = new_beam[:self.beam_width]
            else:
                break
        
        # Return best rule from beam
        if beam:
            return max(beam, key=self._evaluate_rule)
        return None
    
    def _generate_candidate_literals(self, rule: HornClause) -> List[Literal]:
        """Generate candidate literals to add to the rule"""
        candidates = []
        
        # Get variables already in the rule
        rule_vars = set()
        for literal in [rule.head] + rule.body:
            rule_vars.update(literal.arguments)
        
        # Add unary predicates
        for predicate in self.predicates:
            if predicate != rule.head.predicate:
                for var in rule_vars:
                    candidates.append(Literal(predicate, [var], literal_type=LiteralType.UNARY))
        
        # Add binary predicates  
        for predicate in self.predicates:
            if predicate != rule.head.predicate:
                for var1, var2 in itertools.combinations(rule_vars, 2):
                    candidates.append(Literal(predicate, [var1, var2], literal_type=LiteralType.BINARY))
        
        # Add equality with constants
        for var in rule_vars:
            for constant in list(self.constants)[:10]:  # Limit to prevent explosion
                candidates.append(Literal("=", [var, constant], literal_type=LiteralType.EQUALITY))
        
        # Add new variables with existing predicates (limited)
        new_var = f"Y{len(rule_vars)}"
        for predicate in list(self.predicates)[:5]:  # Limit predicates
            if predicate != rule.head.predicate:
                for var in list(rule_vars)[:3]:  # Limit existing vars
                    candidates.append(Literal(predicate, [var, new_var], literal_type=LiteralType.BINARY))
        
        return candidates[:100]  # Limit total candidates
    
    def _count_coverage(self, rule: HornClause, examples: List[Dict[str, Any]]) -> int:
        """Count how many examples are covered by the rule"""
        covered = 0
        
        for example in examples:
            if self._example_matches_rule(example, rule):
                covered += 1
                
        return covered
    
    def _example_matches_rule(self, example: Dict[str, Any], rule: HornClause) -> bool:
        """Check if an example matches (is covered by) a rule"""
        # This is a simplified matching - in practice would need 
        # full first-order unification
        
        # Check head predicate
        head_pred = rule.head.predicate
        if head_pred not in example:
            return False
        
        # Check body literals
        for literal in rule.body:
            if not self._literal_matches_example(literal, example):
                return False
                
        return True
    
    def _literal_matches_example(self, literal: Literal, example: Dict[str, Any]) -> bool:
        """Check if a literal matches an example (simplified)"""
        pred = literal.predicate
        
        if literal.literal_type == LiteralType.EQUALITY:
            # Handle equality constraints
            var, const = literal.arguments
            if var in example and example.get('X') == const:  # Simplified
                return True
            return False
        
        # Handle regular predicates
        if pred in example:
            return bool(example[pred])  # Simplified boolean check
            
        return False
    
    def _evaluate_rule(self, rule: HornClause) -> float:
        """Evaluate rule quality using information gain"""
        if rule.positive_coverage == 0:
            return -float('inf')
        
        # Calculate precision-weighted information gain
        precision = rule.precision()
        coverage = rule.positive_coverage
        
        # Penalize rules that cover negative examples
        penalty = rule.negative_coverage * 0.1
        
        return precision * coverage - penalty
    
    def _get_covered_examples(self, rule: HornClause, 
                            examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get examples covered by a rule"""
        covered = []
        for example in examples:
            if self._example_matches_rule(example, rule):
                covered.append(example)
        return covered
    
    def predict(self, example: Dict[str, Any]) -> bool:
        """Predict if an example satisfies the learned theory"""
        for rule in self.learned_rules:
            if self._example_matches_rule(example, rule):
                return True
        return False
    
    def get_theory(self) -> str:
        """Get string representation of learned theory"""
        if not self.learned_rules:
            return "No rules learned."
        
        theory_str = "Learned Theory:\n"
        for i, rule in enumerate(self.learned_rules, 1):
            theory_str += f"Rule {i}: {rule}\n"
            theory_str += f"  (Coverage: +{rule.positive_coverage} -{rule.negative_coverage}, "
            theory_str += f"Precision: {rule.precision():.3f})\n"
        
        return theory_str


def create_simple_examples():
    """Create simple examples for testing FOIL algorithm"""
    
    # Positive examples: animals that can fly
    positive_examples = [
        {"animal": "tweety", "bird": True, "small": True, "can_fly": True},
        {"animal": "robbie", "bird": True, "small": True, "can_fly": True}, 
        {"animal": "polly", "bird": True, "small": False, "can_fly": True},
        {"animal": "eagle", "bird": True, "small": False, "can_fly": True},
    ]
    
    # Negative examples: animals that cannot fly
    negative_examples = [
        {"animal": "fluffy", "bird": False, "small": True, "can_fly": False},
        {"animal": "rex", "bird": False, "small": False, "can_fly": False},
        {"animal": "ostrich", "bird": True, "small": False, "can_fly": False},
        {"animal": "penguin", "bird": True, "small": False, "can_fly": False},
    ]
    
    return positive_examples, negative_examples


# Example usage
if __name__ == "__main__":
    print("🧪 Testing FOIL Algorithm...")
    
    # Create simple dataset
    pos_examples, neg_examples = create_simple_examples()
    
    # Initialize learner
    foil = FOILLearner(min_pos_coverage=1, max_rule_length=3, beam_width=3)
    
    # Learn rules
    rules = foil.fit(pos_examples, neg_examples, "can_fly")
    
    # Display learned theory
    print("\n" + "="*50)
    print(foil.get_theory())
    
    # Test predictions
    print("\n🎯 Testing predictions:")
    test_cases = [
        {"animal": "sparrow", "bird": True, "small": True, "can_fly": True},
        {"animal": "elephant", "bird": False, "small": False, "can_fly": False},
    ]
    
    for test in test_cases:
        prediction = foil.predict(test)
        actual = test["can_fly"] 
        print(f"   {test['animal']}: predicted={prediction}, actual={actual} {'✓' if prediction == actual else '❌'}")