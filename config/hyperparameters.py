# Tchebycheff
RHO_AUG = 1e-3                        # ρ_aug, range [1e-4, 1e-2]

# EIG
KAPPA = 1.0                            # κ, EIG mix, range [0.5, 2.0]
RHO_AUG_RARE = 0.5                    # ρ_aug_rare, rarity exponent, range [0.3, 0.7]

# Agent σ
GAMMA_SLO = 1.0                       # γ, SLO penalty, range [0.5, 2.0]

# ICP
ALPHA_ICP = 0.05                      # α_ICP, range [0.01, 0.1]
N_ENV_MIN = 3                         # n_env_min, range [3, 5]
N_B = 15                              # n_b, range [10, 30]

# CUSUM (multiples of σ; σ is per-V and computed at runtime)
DELTA_CUSUM_SIGMA = 0.5               # δ_CUSUM multiplier, range [0.3σ, 0.7σ]
H_CUSUM_SIGMA = 4.0                   # h_CUSUM multiplier, range [3σ, 6σ]

# FourQuadrant
SLO_TOLERANCE = 0.15                  # slo_tolerance, range [0.10, 0.20]

# ConfidenceService — hand-set lookup tables
DELTA_C_EDGE_LOOKUP = None            # Δc_edge_lookup, see confidence table
DELTA_C_MECH_LOOKUP = None            # Δc_mech_lookup, see confidence table

# DRO
EPSILON_DRO_INIT = 0.15               # ε_DRO^(0), range [0.05, 0.30]
DRO_TARGET_COVERAGE = 0.90           # dro_target_coverage, range [0.85, 0.95]
ETA_EPSILON_DRO = 0.05               # η_ε_DRO, range [0.02, 0.10]

# SwitchCost
TAU_AB_HOURS = 1 / 12                 # τ_ab, canary test duration = 5 min, range [3 min, 15 min]
KILL_COST_DEFAULT = 0.02              # kill_cost_default ($), range [$0.01, $0.10]
LAMBDA_RISK = 50.0                    # λ_risk ($), range [$10, $200]

# SlowLoop
ETA_W = 0.10                          # η_w, Tchebycheff weights EMA, range [0.05, 0.20]
ETA_LAMBDA = 0.05                     # η_λ
ETA_BETA = 0.10                       # η_β
RHO_STAR_SWIT_INIT = 0.20            # ρ*_swit initial (annealed → 0.05)
RHO_STAR_SWIT_FINAL = 0.05           # ρ*_swit final
RHO_STAR_SLOPE_INIT = 0.20           # ρ*_slope initial (annealed → 0.02)
RHO_STAR_SLOPE_FINAL = 0.02          # ρ*_slope final
B_MIN = 1                             # B_min
B_MAX = 10                            # B_max

# Regret
W_REGRET = 20                         # W_regret, regret window (ticks)
W_Q1 = 20                             # W_q1, Q1 rate window (ticks)

# Agent
K_P = 1                               # K_p, plan samples per tick (v0=1, v2=>1)
K_MAX = 64                            # K_max, tool calls per ReAct trajectory, range [32, 128]
TEMPERATURE_PER_PHASE = (0.1, 0.5, 0.7, 0.2, 0.4)  # temperature per FSM phase
