class RegretCalculator:
    def compute_q1_rate(self, evidence_store, tick, window):
        # Placeholder: return Q1 rate in [0,1] over window
        pass

    def compute_q1_rate_per_mechanism(self, evidence_store, mechanism_id, window):
        # Placeholder: return Q1 rate for a mechanism, or None if insufficient data
        pass

    def compute_q1_rate_per_env(self, evidence_store, env, window):
        # Placeholder: return Q1 rate for an environment, or None if insufficient data
        pass

    def compute_inst_regret(self, tick, evidence_store, window):
        # Placeholder: return instantaneous regret in [0,1]
        pass

    def compute_cum_regret(self, ticks, evidence_store, window):
        # Placeholder: return cumulative regret in [0,1]
        pass

    def compute_regret_slope(self, tick, evidence_store, window):
        # Placeholder: return slope of regret over window
        pass

    def compute_outcome_regret(self, evidence_store, ticks):
        # Placeholder: return outcome-based regret in [0,1]
        pass
