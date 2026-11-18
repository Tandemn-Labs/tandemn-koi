The resource map needs to change and it's functions also need to change.

The definition of new resource map can be found here -> https://github.com/Tandemn-Labs/tdcp/blob/master/proto/tandemn/resourcemap/v1alpha1/resource_map.proto. This is the PER CLUSTER resource map. The GLOBAL View of Cluster Resources are present here -> https://github.com/Tandemn-Labs/tdcp/blob/master/proto/tandemn/resourcemap/v1alpha1/global_resource_map.proto. The wordings and schema is not fully fixed but the structure is still very much fixed.

Also REMEMBER - the GLOBAL RESOURCE MAP contains multiple clusters, but they all are GOVERNED By the same agent and agent tools (koi - brain). So, always remember, there is ONE KOI FOR ALL CLUSTERS, and not ONE KOI PER CLUSTER. Ranks and Rank IDs are NOT UNIQUE to each cluster gfiven that rank_id is created by Koi itself.

The whole idea is that the resource map is a tree based structure where there are two views -
a) Node View: Which shows all the nodes and their occupancy grid
b) Job View: Which shows the job view.

The design choice is that there is a Resource Map Service (not implemented in resource_map.py) and that service acts as a "RECIEVER" of the resource map from the Operators. resource_map.py is a CLIENT that will run a function called get_latest_resource_map() (keep it unimplemented), but the idea is that it will do an RPC call to a Resource Map Service and the answer returned is the Global Resource Map proto view.

The responsibilities of the resource_map.py are these
a) Get the latest Resource Map from the RM Service
b) Convert that Proto into native python classes.
c) Have Helper Functions that help Koi's Agent/Agent.py or any other downstream user of Resource Map interact with the resource map.

The live planning path operates on the native Proto-derived map. Review changes after commit `ed86c8cfa03f29a0b802d139277d2265447af0c3` carefully.

## PR boundary and agreed contracts

**This PR:** preserve and persist the complete GlobalResourceMap per tick in memory; use one frozen map across all clusters for one Koi; expose resource facts; implement projected capacity, pricing, and plan-target persistence. NodeOccupancy is authoritative for actual occupancy regardless of pod phase. A resource group is identified by `(cluster_id, pool_id, node_type_id)`; identical-looking groups in different clusters remain distinct. Convert directly from the proto assumign that it is validated. Avoid a proto → JSON → dictionary → class pipeline.


`get_latest_resource_map()` remains unimplemented. Operator ingestion, the RM Service receiver/RPC, submission transport, metrics writing, and operator execution are outside this PR.

**Next PR:** migrate Agent Tools to consume the per-cluster raw resource records and apply its existing planning policy. Validator C5/C6 exact-group validation and downstream Deployment X adaptation are also downstream work. Telemetry migration and `observed_replica_liveness()` are not part of this PR. Candidate construction that chooses target IDs is downstream; this PR must preserve any supplied IDs through plan copying/serialization and persist them to the placement row. The scope of RPC Call failing remains out of scope for this PR. WHat might happen in the future is that we will preserve the 'observed_at' time for a given cluster snapshot in memory and re-use last tick's resource map, but all that will be WITHIN the scope of the resource map serbice and this PR should NOT WORRY about it. Protobuf validation is also NOT in scope for this PR.

Here are some functions we are planning by the business functionality -

a) Snapshot and Resource Access
   - snapshot_cluster_state(tick): freeze the cluster_state and attach every job's intent and context. (I am not sure if it is needed).
   - resource_group(cluster_id, pool_id, node_type_id): Locate the group Record and View it
   - resources_summary(): Return a per-cluster index of the canonical GlobalResourceMap records. Do not flatten or merge clusters into legacy environment keys; one Koi agent governs the whole global map. There are sub-functions within this function for example
    - get_running_jobs(): Canonical Sources of the Jobs Running should come from the Resource Map. A new resource map leads to editing the job statuses of all the jobs. For example: in the last tick - a job 'job_abc' was 'queued', but the resource map showed it's updated status as 'running', so that job's status in the database should change from 'queued' to 'running'. So the state changes are 'queued' -> 'pending' -> 'running' -> 'finished'.
    - get_waiting_jobs(): The Canonical Source is our database. Waiting Jobs are Jobs with the 'queued' status. It may or may not be available in the live resource map, so direct it from the Database.
    - get_submitted_jobs(): It was an alias for the get_waiting_jobs(), it is no longer needed in the future version.


b) Resource Facts and Availability
   - node_headroom(...): Physical Headroom each node has from NodeOccupancy
   - get_avail_capacity(...): expose physical headroom, autoscaling reservations, and pending demand as separate facts; do not debit pod allocations twice or silently prioritize pending demand.
   - pool_capacity(...): expose total capacity, NodeOccupancy usage, autoscaling reservations, and remaining capacity separately, plus available allocation units = total capacity minus NodeOccupancy usage and autoscaling reservations. Pending demand remains separate for Agent Tools policy.
   - rank_allocation_summary(rank): Describe the rank's footprint
   - rank_capacity_per_replica(rank) / rank_capacity_footprint(rank) — per-replica and total requested footprint.
   - requested_capacity(plan) — describe a proposal’s demands without modifying observed state.

c) Simulation, checking, and pricing
   - simulate_future_resources(plan) — project Place/Swap ladders against a copy of the map. Use each rank's environment, instance/GPU configuration, and replica count to project capacity. For each replica, first best-effort pack onto a partially occupied existing node with enough headroom; if none fits, place it on a fresh, already-existing idle node. Do not fail just because co-location failed, dedicate a node when an occupied node fits, or add nodes absent from the map. Project the completed plan: apply removals/releases before additions; Koi does not submit overlapping old/new Swap placements. A matching cluster may be chosen freely where cluster identity does not affect scoring. Simulation need not assign Kubernetes node or pod IDs and must not mutate observed occupancy.
   - check_resource_feasibility(plan) — check the proposal against that structure and existing commitments.
   - switch_pricing_map() — attribute cost by exact target and occupied accelerators. For reserved capacity, charge each job for the GPUs it occupies, including its autoscaling reservation; do not amortize already-paid capacity. For on-demand/spot nodes shared by jobs, amortize node price among those jobs in proportion to their occupied accelerators.

Agent Tools and Validator consume the per-cluster map in the next PR; this PR establishes the native-map and helper contracts without moving allocation priority or selection policy into resource_map.py.

Accounting Problems in the Legacy Resource Map:

The Legacy Tandemn Store era Resource Map had some problems around accounting for resources
1 - Detection of Phantom IDs: _phantom_chain_ids was the code that was made to detect phantom chains from telemetry, given the legacy path of resource map was more of a STATIC resource map which did not differentiate between running replicas and reserved for autoscaling replicas. This behavior needs to end. If Koi's agent makes a decision to reserve 5 GPUs for one job and it's currently running only on 2 replicas and the rest 3 are still reserved for autoscaling  - that info should coming from the New Live Resource Map and not from metrics.
2 - The biggest problem we had in the legacy tandemn store based resource map was that it had a static resource map and koi was internally tracking the state of the resource map. Even though that was a good practice for planning and we need to keep that capability, the problem was that, we did not let the cluster actually tell us about the live state of the cluster. Kubernetes does bin packing and multiple jobs can run in a cluster. What we were doing in legacy version was - in the internal resource_map.py, for example, when we had two jobs, we were reserving TWO NODES separately and koi's next turns would think that two nodes were fully occupied. What ended up happening downstream was that K8s saw the two pods and decided to place them onto single node (as they could satisfy the requirements.). That lead to a SPLIT BRAIN in scheduling. Internal Koi's RM thought 2 nodes were occupied - but only one was actually occupied. Even if the agent and agent tools are doing the legacy scheduling- its ok, not a part of the current PR Migration, but it's intent should atleast not be on the resource_map.py side.

After this is done - delete all the dead code in resource_map.py that serves no purpose.
