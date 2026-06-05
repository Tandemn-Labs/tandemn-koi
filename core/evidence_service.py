class EvidenceService:
    def append_row(self, residual_history):
        # Placeholder: append an EvidenceRow
        # uses write_theory_narrative() internally
        return residual_history, theory_blob

    def get_row(self, job_id, rank_id, chain_id):
        # Placeholder: return EvidenceRow for (job_id, rank_id, chain_id)
        pass

    def get_rows_in_window(self, window):
        # Placeholder: return all EvidenceRows within the given time window
        pass

    def get_rows_for_edge(self, edge_id, limit=None):
        # Placeholder: return EvidenceRows for all ranks whose mechanisms contain this edge
        pass

    def get_rows_for_mechanism(self, mechanism_id, limit=None):
        # Placeholder: return EvidenceRows for all ranks with this mechanism attached
        pass

    def get_rows_for_job(self, job_id):
        # Placeholder: return all EvidenceRows for a job
        pass

    def get_rows_for_environment(self, envs):
        # Placeholder: return EvidenceRows matching the given environments
        pass

    def get_recently_decided(self, window):
        # Placeholder: return EvidenceRows for recently decided rows in window
        pass

    def count_visits_per_edge(self, edge_id):
        # Placeholder: return visit count for an edge
        pass

    def count_envs_per_edge(self, edge_id):
        # Placeholder: return (envs, count) for an edge
        pass

    def last_touched_per_edge(self, edge_id):
        # Placeholder: return the tick when edge was last touched
        pass

    def q3_rate_window(self, edge_id, window):
        # Placeholder: return Q3 rate for an edge within a time window
        pass

    def write_theory_narrative(self, evidence_row_id, narrative):
        # Placeholder: write a narrative to an EvidenceRow, return Bool
        pass

    def aggregate(self):
        # Placeholder: aggregate evidence (dashboard use)
        pass
