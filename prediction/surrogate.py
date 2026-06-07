class SurrogatePrediction:
    def compose_prediction(self, job_config, mechanism, job_features):
        # Master Function - Given JobConfig, Mechanism, and Workload characteristics, call the surrogate stack to return the predictions
        # Inputs: JobConfig(ladder), Mechanism, JobFeatures[Workload, Hardware, WorkloadType, Environment]
        # Outputs: y_hat, v_hat, dro_band 
        # let dro_band be none for now, till we figure out how that is done
        pass

    def get_model_config(self, model_id):
        # Fetch model architecture from Huggingface or a similar place
        # Inputs: model_id
        # Outputs: config.json
        pass

    def get_env_row(self, job_features):
        # Fetch the Env and the cloud we want for the prediction
        # Inputs: JobFeatures[Environment, Hardware]
        # Outputs: EnvVector
        pass

    def create_eligibility_mask(self, X_list, V_list, Y_list):
        # Scope X, V, and Y to eligible entries for the surrogate stack
        # Inputs: List[X], List[V], List[Y]
        # Outputs: EligibilityMask
        pass

    def fetch_cloud_prices(self, env_vector):
        # Fetch real-time per-hour cost of compute resources
        # Inputs: EnvVector
        # Outputs: Per Hour Pricing
        pass

    def set_prediction_scope(self, candidate_graph, eligibility_mask, job_features):
        # Extract eligible X, V, and Y for this run based on mask and features
        # Inputs: CandidateGraph, EligibilityMask, JobFeatures
        # Outputs: EligibleX, EligibleV, EligibleY
        pass

    def build_surrogate_inputs(self, eligible_X, job_config, model_config, env_vector, price_vector):
        # Build inputs for selected surrogate
        # Inputs: EligibleX, JobConfig, ModelConfig, EnvVector, PriceVector
        # Outputs: SurrogateInput
        pass

    def run_surrogate(self, surrogate_input, method, accumulate_logic="average"):
        # Run the surrogate model.
        # Inputs: SurrogateInput, Method=List[DynoSim, LLMSimulator, etc], accumulate_logic: average,llm decides
        # Outputs: y_hat, v_hat
        pass