"""
📋   Init  
============

🔬 Research Foundation:
======================
Based on inductive logic programming research:
- Quinlan, J.R. (1990). "Learning Logical Definitions from Relations"
- Muggleton, S. & De Raedt, L. (1994). "Inductive Logic Programming: Theory and Methods"
- Lavrac, N. & Dzeroski, S. (1994). "Inductive Logic Programming: Techniques and Applications"
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
🔍 PREDICATE SYSTEM MODULES - Logical Vocabulary Management
===========================================================

Lightweight modular predicate system for organizing logical vocabularies in ILP systems.

🧠 Inductive Logic Programming Library - Made possible by Benedict Chen
   benedict@benedictchen.com
   Support his work: 🍺 Buy him a beer: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Sponsor: https://github.com/sponsors/benedictchen

📚 Research Foundation:
- Muggleton, S. & De Raedt, L. (1994). "Inductive Logic Programming: Theory and Methods."
- Established framework for predicate management in ILP systems

🎯 ELI5 Explanation:
Think of predicates like the vocabulary words in a language. Just as you need
to know what "parent", "older", "human" mean to understand family relationships,
ILP systems need to organize and manage their logical vocabulary.

This module is like a smart dictionary that keeps track of what words (predicates)
are available and how they relate to each other.

🔧 Core Components:
• CorePredicateSystem: Basic predicate management
• PredicateDefinition: Individual predicate specifications

🙏 Support This Work:
Your support makes continued development of research-accurate ILP tools possible!
"""

from .predicate_core import CorePredicateSystem, PredicateDefinition

__all__ = ['CorePredicateSystem', 'PredicateDefinition']

print("""
💰 MODULE SUPPORT - Made possible by Benedict Chen
   ]8;;mailto:benedict@benedictchen.com\benedict@benedictchen.com]8;;\

💰 PLEASE DONATE! Your support keeps this research alive! 💰
   🔗 ]8;;https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS\💳 CLICK HERE TO DONATE VIA PAYPAL]8;;\
   ❤️ ]8;;https://github.com/sponsors/benedictchen\💖 SPONSOR ON GITHUB]8;;\

   ☕ Buy me a coffee → 🍺 Buy me a beer → 🏎️ Buy me a Lamborghini → ✈️ Buy me a private jet!
   (Start small, dream big! Every donation helps! 😄)
""")
