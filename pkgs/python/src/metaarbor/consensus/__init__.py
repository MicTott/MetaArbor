"""MetaArbor-Consensus (EXPERIMENTAL, `consensus-prototype` branch only).

Consumes the frozen pairwise MetaArbor API; Walk and Transport are never
altered. See DESIGN.md in this package for the specification, isolation
rules, prespecified predictions, and validation gates.
"""
from .eligibility import (call, call_v2, eligible_v1, eligible_v15,
                          p_detect, p_detect_posterior,
                          prevalence_lower, prevalence_posterior,
                          support)
from .poset import compatible, pair_relation, relation
from .simulate import latent_tree, scenario, simulate_donors
