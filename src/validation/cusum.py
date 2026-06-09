class Cusum:
    def cusum_per_mechanism(self, mechanism, v_actual_traj, v_hat_traj, H, residual_table):
        # Placeholder: return Enum[Matched, Diverged] for the mechanism
        pass

    def cusum_per_v(self, v_observed, v_predicted, residual_table):
        # Placeholder: return (Direction, Bool[CrossThreshold], tick_for_CrossThreshold)
        pass

    def compute_cusum_statistic(self, residuals, residual_table, edge_table):
        # Placeholder: return (S_plus, S_minus, CrossThreshold)
        pass

    def update_cusum_from_history(self, historical_residuals_per_v):
        # Placeholder: return Dict[V -> (DeltaTable, EdgeTable)]
        pass

    def cusum_params_per_v(self, V, historic_residuals):
        # Placeholder: return (DeltaTable, EdgeTable) for V
        pass
