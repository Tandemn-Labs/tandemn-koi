The resource map needs to change and it's functions also need to change.

The definition of new resource map can be found here -> https://github.com/Tandemn-Labs/tdcp/blob/master/proto/tandemn/resourcemap/v1alpha1/resource_map.proto. This is the PER CLUSTER resource map. The GLOBAL View of Cluster Resources are present here -> https://github.com/Tandemn-Labs/tdcp/blob/master/proto/tandemn/resourcemap/v1alpha1/global_resource_map.proto. The wordings and schema is not fully fixed but the structure is still very much fixed.

Also REMEMBER - the GLOBAL RESOURCE MAP contains multiple clusters, but they all are GOVERNED By the same agent and agent tools (koi - brain). So, always remember, there is ONE KOI FOR ALL CLUSTERS, and not ONE KOI PER CLUSTER.

The whole idea is that the resource map is a tree based structure where there are two views -
a) Node View: Which shows all the nodes and their occupancy grid
b) Job View: Which shows the job view.

The design choice is that there is a Resource Map Service (not implemented in resource_map.py) and that service acts as a "RECIEVER" of the resource map from the Operators. resource_map.py is a CLIENT that will run a function called get_latest_resource_map() (keep it unimplemented), but the idea is that it will do an RPC call to a Resource Map Service and the answer returned is the Global Resource Map proto view.

The responsibilities of the resource_map.py are these
a) Get the latest Resource Map from the RM Service
b) Convert that Proto into native python classes.
c) Save the Resource Map into Database for observability / History Accounting.
d) Have Helper Functions that help Koi's Agent/Agent.py or any other downstream user of Resource Map interact with the resource map.

a is straightforward. b might take some time, but c is the main one. Helper functions are to be designed in a way that agent/agent_tools are unchanged and whatever data that they require, it will be processed from the live resource map and provided. Given that it is a migration - most of the code we changed AFTER commit ed86c8cfa03f29a0b802d139277d2265447af0c3 is to be properly checked. We will NOT READ the resource map from the database table and then convert it into JSON and do operations on it like legacy resource map. The whole point of Proto format is that we can convert it into Native Python Classes and operate on them.

Here are some functions we are planning by the business functionality -

a) Snapshot and Resource Access
   - read_resource_map_snapshot(tick): read the exact resource map for that tick
   - snapshot_cluster_state(tick): freeze the cluster_state and attach every job's intent and context. (I am not sure if it is needed).
   - resource_group(cluster_id, pool_id, node_type_id): Locate the group Record and View it
   - resource_summary(): Expose the summary as it was shown in the legacy for the agent and agent tools.

b) Resource Facts and Availability
   - node_headroom(...): Physical Headroom each node has from NodeOccupancy
   - get_avail_capacity(...): capacity available to new proposals, accounting for already running pods and for committed pods/capacity reserved (but even un-used / inactive rn ) for autoscaling.
   - pool_capacity(...): total capacity for that NodePool
   - rank_allocation_summary(rank): Describe the rank's footprint
   - rank_capacity_per_replica(rank) / rank_capacity_footprint(rank) — per-replica and total requested footprint.
   - requested_capacity(plan) — describe a proposal’s demands without modifying observed state.

c) Simulation, checking, and pricing
   - simulate_future_resources(plan) — simulate against a copy of the existing resource map, with the new plan that has changed placements. The plan’s place and swap actions carry ladders; each rank supplies its environment, configuration (including instance type and GPU shape), and replica count. Those fields let the Resource Map calculate the capacity the plan would reserve. The new simulate would look at the environment, and can provide the "projected" remaining capacity by this policy - best effort colocation into existing nodes. If a new rank is created, try to colocate with existing ranks on existing nodes, if not, put it on a new node. Do not have the policy of ONE NODE dedicated per placement. For deletions, it is straightforward. Do not worry about the exact cluster_id, any cluster being selected is ok given that it does not change the downstream cluster_score calculations. At the same time, the exact kubernetes ID for that pod does not need to be a concern.
   - check_resource_feasibility(plan) — check the proposal against that structure and existing commitments.
   - switch_pricing_map() — accelerator-proportional pricing by exact target. This is a bit nuanced
        - if a job is running on reserved GPUs, the cost per job is only the cost totalled by the number of GPUs it is occupying (both running and reserved for autoscaling). Given that Reserved GPUs are already paid for - amortization does not really make sense.
        - if a job is running on on-demand / spot GPUs- the cost per job is amortized between the number of jobs running across the same nodes. If two jobs are sharing the same on demand instance, the cost should be amortized between them in a proportional way.

The goal is to NOT REALLY DISTURB Agent and Agent Tools, and move things around for it in resource_map.py. It is not a HARD AND FAST rule- obviously if something is WRONG In resource map.

Accounting Problems in the Legacy Resource Map:

The Legacy Tandemn Store era Resource Map had some problems around accounting for resources
1 - Detection of Phantom IDs: _phantom_chain_ids was the code that was made to detect phantom chains from telemetry, given the legacy path of resource map was more of a STATIC resource map which did not differentiate between running replicas and reserved for autoscaling replicas. This behavior needs to end. If Koi's agent makes a decision to reserve 5 GPUs for one job and it's currently running only on 2 replicas and the rest 3 are still reserved for autoscaling  - that info should coming from the New Live Resource Map and not from metrics.
2 - The biggest problem we had in the legacy tandemn store based resource map was that it had a static resource map and koi was internally tracking the state of the resource map. Even though that was a good practice for planning and we need to keep that capability, the problem was that, we did not let the cluster actually tell us about the live state of the cluster. Kubernetes does bin packing and multiple jobs can run in a cluster. What we were doing in legacy version was - in the internal resource_map.py, for example, when we had two jobs, we were reserving TWO NODES separately and koi's next turns would think that two nodes were fully occupied. What ended up happening downstream was that K8s saw the two pods and decided to place them onto single node (as they could satisfy the requirements.). That lead to a SPLIT BRAIN in scheduling. Internal Koi's RM thought 2 nodes were occupied - but only one was actually occupied. Even if the agent and agent tools are doing the legacy scheduling- its ok, not a part of the current PR Migration, but it's intent should atleast not be on the resource_map.py side.

After this is done - delete all the dead code in resource_map.py that serves no purpose.
