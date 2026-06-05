def get_mechanism(mechanism_id):
    # Placeholder: return Mechanism for the given id
    pass

def add_mechanism(Mechanism):
    # Placeholder: add a Mechanism and return its assigned mechanism_id
    pass

def get_usable_mechanism(mechanisms):
    # Placeholder: return the subset of mechanisms that are currently usable
    pass

def get_archived_mechanism(mechanisms):
    # Placeholder: return the subset of mechanisms that are archived
    pass

def filter_by_scope(job_features):
    # Placeholder: filter mechanisms by JobFeatures (workload, hardware, workload type)
    pass

def archive_mechanism(mechanism_id, reason):
    # Placeholder: archive a mechanism and return success bool
    pass

def is_duplicate_mechanism(mechanism_id):
    # Placeholder: check if mechanism is a duplicate, return (Bool, Optional[mechanism_id])
    pass

def get_mechanism_containing_edge(edge_id):
    # Placeholder: return all mechanisms that contain the given edge
    pass

def get_edges_from_mechanism(mechanism_id):
    # Placeholder: return all edges belonging to a mechanism
    pass

def does_scope_match(Scope, job_features):
    # Placeholder: check whether a scope matches given job features
    pass

def val_mechanism(Mechanism):
    # Placeholder: validate a mechanism, return (Bool, Optional[List[Violation]])
    pass
