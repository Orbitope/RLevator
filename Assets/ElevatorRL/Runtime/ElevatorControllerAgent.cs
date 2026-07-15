using UnityEngine;
using UnityEngine.Rendering;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace ElevatorRL
{
    /// <summary>
    /// Single controller agent. Owns one headless Building, ticks it every
    /// FixedUpdate, and requests a joint decision on a fixed sim-time cadence. The
    /// MultiDiscrete head emits one branch (size 6) per car at the FIXED policy width
    /// (cfg.numElevators); out-of-service cars are masked to NOOP, so one trained
    /// policy serves any active fleet size between minActiveElevators and numElevators.
    ///
    /// Requires a BehaviorParameters component on the same GameObject. ActionSpec and
    /// VectorObservationSize are configured from the assets in Initialize so the inspector
    /// can stay untouched.
    /// </summary>
    /// <summary>Action-space variant (EXPERIMENT_PLAN.md §7 / E9).</summary>
    public enum ActionSpaceMode
    {
        /// <summary>AS0: one size-6 branch per car (noop/up/down/board-up/board-down/unload),
        /// decided every eligible tick.</summary>
        Primitive,
        /// <summary>AS1: one size-numFloors branch per car (commit to a target floor); a
        /// deterministic sub-controller drives the car there and the policy is re-queried only on
        /// arrival (semi-MDP / options).</summary>
        TargetFloor,
    }

    [RequireComponent(typeof(BehaviorParameters))]
    public sealed class ElevatorControllerAgent : Agent
    {
        [Header("Environment assets")]
        public BuildingConfig buildingConfig;
        public RewardConfig rewardConfig;
        public ObservationConfig observationConfig;
        public TrafficConfig trafficConfig;

        [Header("Action space (E9)")]
        [Tooltip("Primitive = per-tick low-level actions (AS0). TargetFloor = commit-to-floor " +
            "semi-MDP, re-queried only on arrival (AS1). Changing this changes the action branch " +
            "sizes, so retrain — a policy trained under one mode is not valid under the other.")]
        public ActionSpaceMode actionSpace = ActionSpaceMode.Primitive;

        [Header("Episode")]
        [Tooltip("Decisions per episode before truncation (continuing task). 0 = never truncate here (rely on Max Step).")]
        public int episodeDecisionLimit = 2048;
        public int seed = 1;

        [Header("Optional read-only view")]
        public ElevatorRenderer view;

        public Building Sim => _b;

        Building _b;
        // Runtime clone of trafficConfig so E12 can override the traffic pattern / day-cycle per
        // training run via ML-Agents environment parameters (traffic_pattern, use_day_cycle) WITHOUT
        // editing the shared asset or rebuilding — one headless build per size then serves every
        // pattern. See OnEpisodeBegin and EXPERIMENT_PLAN.md E12.
        TrafficConfig _traffic;
        bool _loggedTraffic;
        float _clock;
        int _decisions;

        // AS1 (TargetFloor) per-car commitment state
        int[]  _target;
        bool[] _committed;

        public override void Initialize()
        {
            // Clone so per-run env-parameter overrides (traffic_pattern/use_day_cycle) don't mutate
            // the shared TrafficConfig asset. Falls back to the asset's baked values when unset.
            _traffic = trafficConfig != null ? Instantiate(trafficConfig) : null;
            _b = new Building(buildingConfig, rewardConfig, observationConfig, _traffic, seed);

            int E = buildingConfig.numElevators;
            _target = new int[E];
            _committed = new bool[E];

            // AS0: 6 primitive actions per car. AS1: one target-floor choice (0..numFloors-1) per car.
            int branchSize = actionSpace == ActionSpaceMode.TargetFloor ? buildingConfig.numFloors : 6;
            var bp = GetComponent<BehaviorParameters>();
            var branches = new int[E];
            for (int i = 0; i < branches.Length; i++) branches[i] = branchSize;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeDiscrete(branches);
            bp.BrainParameters.VectorObservationSize = _b.ObservationSize();
        }

        public override void OnEpisodeBegin()
        {
            // Curriculum hook: set "active_elevators" in your yaml to pin the fleet size,
            // or leave it unset to let the config randomize it.
            int forced = -1;
            var ep = Academy.Instance.EnvironmentParameters;
            float v = ep.GetWithDefault("active_elevators", -1f);
            if (v > 0f) forced = Mathf.RoundToInt(v);

            // E12: per-run traffic override. traffic_pattern = 0..4 (UpPeak/DownPeak/Lunch/Midday/
            // Uniform); use_day_cycle = 0/1. Either < 0 leaves the asset's baked value untouched.
            if (_traffic != null)
            {
                float pat = ep.GetWithDefault("traffic_pattern", -1f);
                if (pat >= 0f) _traffic.defaultPattern = (TrafficPattern)Mathf.RoundToInt(pat);
                float day = ep.GetWithDefault("use_day_cycle", -1f);
                if (day >= 0f) _traffic.useDayCycle = day > 0.5f;

                if (!_loggedTraffic)
                {
                    _loggedTraffic = true;
                    Debug.Log($"[E12] traffic override active: pattern={_traffic.defaultPattern} " +
                              $"useDayCycle={_traffic.useDayCycle} (env params traffic_pattern={pat}, use_day_cycle={day})");
                }
            }

            _b.Reset(forced);
            _clock = 0f;
            _decisions = 0;
            for (int i = 0; i < _committed.Length; i++) { _committed[i] = false; _target[i] = 0; }
        }

        void FixedUpdate()
        {
            if (_b == null) return;
            float dt = Time.fixedDeltaTime;
            _b.Tick(dt);

            // AS1: every tick, drive each committed car one primitive step toward its target via
            // the deterministic sub-controller (no policy involvement). ApplyAction/ResolveToward
            // both no-op while a car is mid-travel or in a door cycle, so this only fires when a
            // committed car is idle at a floor and needs its next step re-issued.
            if (actionSpace == ActionSpaceMode.TargetFloor)
                for (int i = 0; i < _b.cars.Length; i++)
                    if (_committed[i])
                        _b.ApplyAction(i, TargetFloorControl.ResolveTowardTarget(_b, i, _target[i]));

            _clock += dt;
            if (_clock >= buildingConfig.decisionInterval)
            {
                _clock -= buildingConfig.decisionInterval;
                _b.StepServiceChanges();

                if (actionSpace == ActionSpaceMode.TargetFloor)
                {
                    // release commitments that have been fulfilled (arrived + nothing left to do)
                    // so those cars are re-queried; leave in-flight cars committed.
                    for (int i = 0; i < _b.cars.Length; i++)
                        if (_committed[i] && TargetFloorControl.CommitmentFulfilled(_b, i, _target[i]))
                            _committed[i] = false;

                    // AS1 is a semi-MDP: only ask the policy when at least one car actually needs a
                    // new target (idle, in service, uncommitted) — decisions are sparse (one per
                    // dispatch), which is the whole point of this action space.
                    if (AnyCarNeedsTarget()) RequestDecision();
                }
                else
                {
                    // AS0: skip the decision entirely if every car is mid-travel/door-cycle (all 5
                    // non-NOOP actions masked for every car this tick) — with floorTravelTime
                    // (1.6s default) several times decisionInterval (0.5s), most fixed-cadence ticks
                    // would otherwise produce a fully-masked, wasted OnActionReceived call. This
                    // removes wasted decisions, not the up-to-decisionInterval latency between a car
                    // becoming idle and the policy next acting on it. Also means episodeDecisionLimit
                    // corresponds to a fleet/traffic-dependent amount of sim time.
                    if (AnyCarAtFloor()) RequestDecision();
                }
            }

            // Skip visualization entirely under -nographics (headless training builds): the
            // required shader isn't stripped-in for a graphics-less build (Shader.Find returns
            // null there), and there's no point paying render cost across N parallel workers anyway.
            if (view != null && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
                view.Mirror(_b);
        }

        bool AnyCarAtFloor()
        {
            for (int i = 0; i < _b.cars.Length; i++)
            {
                var c = _b.cars[i];
                if (c.inService && c.AtFloor) return true;
            }
            return false;
        }

        bool AnyCarNeedsTarget()
        {
            for (int i = 0; i < _b.cars.Length; i++)
            {
                var c = _b.cars[i];
                if (c.inService && c.AtFloor && !_committed[i]) return true;
            }
            return false;
        }

        public override void CollectObservations(VectorSensor sensor) => _b.WriteObservation(sensor);

        public override void OnActionReceived(ActionBuffers actions)
        {
            var d = actions.DiscreteActions;

            if (actionSpace == ActionSpaceMode.TargetFloor)
            {
                // Adopt a new target only for cars that are actually asking (idle, in service,
                // uncommitted). In-flight/committed cars keep their target — their emitted branch is
                // masked to a single option and ignored here. The per-tick driver in FixedUpdate is
                // what actually moves cars; this method only updates the commitment target.
                for (int i = 0; i < buildingConfig.numElevators; i++)
                {
                    var c = _b.cars[i];
                    if (c.inService && c.AtFloor && !_committed[i])
                    {
                        _target[i] = Mathf.Clamp(d[i], c.minFloor, c.maxFloor);
                        _committed[i] = true;
                    }
                }
            }
            else
            {
                for (int i = 0; i < buildingConfig.numElevators; i++)
                    _b.ApplyAction(i, d[i]);
            }

            // Reward accumulated since the previous decision. In AS1 decisions are sparse, so each
            // call captures all reward earned while the last chosen targets were being executed —
            // correct semi-MDP credit attribution.
            AddReward(_b.CollectReward());

            _decisions++;
            // This is a continuing task with no genuine terminal state — the decision-limit
            // cutoff is a truncation, not a termination. EndEpisode() would tell PPO the final
            // state's value is 0 (false) and bias the value function at exactly the truncation
            // states; EpisodeInterrupted() correctly bootstraps the value instead.
            if (episodeDecisionLimit > 0 && _decisions >= episodeDecisionLimit) EpisodeInterrupted();
        }

        public override void WriteDiscreteActionMask(IDiscreteActionMask mask)
        {
            if (actionSpace == ActionSpaceMode.TargetFloor)
            {
                int F = buildingConfig.numFloors;
                for (int i = 0; i < buildingConfig.numElevators; i++)
                {
                    var c = _b.cars[i];
                    bool needsTarget = c.inService && c.AtFloor && !_committed[i];

                    if (!needsTarget)
                    {
                        // Committed/in-flight or OOS car: lock the branch to a single legal option
                        // (its current target, or floor 0) so the policy's emitted value is a no-op.
                        int keep = Mathf.Clamp(_committed[i] ? _target[i] : 0, 0, F - 1);
                        for (int f = 0; f < F; f++) mask.SetActionEnabled(i, f, f == keep);
                    }
                    else
                    {
                        // Asking car: only floors within its service range are legal targets
                        // (current floor stays legal = "hold").
                        for (int f = 0; f < F; f++)
                            mask.SetActionEnabled(i, f, f >= c.minFloor && f <= c.maxFloor);
                    }
                }
                return;
            }

            for (int i = 0; i < buildingConfig.numElevators; i++)
            {
                var c = _b.cars[i];

                // out of service, or mid-travel / door cycle: only NOOP is legal
                if (!c.inService || !c.AtFloor)
                {
                    for (int a = 1; a <= 5; a++) mask.SetActionEnabled(i, a, false);
                    continue;
                }

                if (c.Floor >= c.maxFloor) mask.SetActionEnabled(i, 1, false); // can't go up
                if (c.Floor <= c.minFloor) mask.SetActionEnabled(i, 2, false); // can't go down
                if (_b.upQ[c.Floor].Count == 0) mask.SetActionEnabled(i, 3, false);   // nothing to board up
                if (_b.downQ[c.Floor].Count == 0) mask.SetActionEnabled(i, 4, false); // nothing to board down
                if (!c.WantsFloor(c.Floor)) mask.SetActionEnabled(i, 5, false);       // nobody to drop here
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var d = actionsOut.DiscreteActions;
            if (actionSpace == ActionSpaceMode.TargetFloor)
            {
                var targets = TargetFloorControl.LookTargets(_b);
                for (int i = 0; i < targets.Length; i++) d[i] = targets[i];
            }
            else
            {
                var plan = ElevatorHeuristics.CollectiveLook(_b);
                for (int i = 0; i < plan.Length; i++) d[i] = plan[i];
            }
        }
    }
}
