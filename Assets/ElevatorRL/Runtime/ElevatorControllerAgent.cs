using UnityEngine;
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
    [RequireComponent(typeof(BehaviorParameters))]
    public sealed class ElevatorControllerAgent : Agent
    {
        [Header("Environment assets")]
        public BuildingConfig buildingConfig;
        public RewardConfig rewardConfig;
        public ObservationConfig observationConfig;
        public TrafficConfig trafficConfig;

        [Header("Episode")]
        [Tooltip("Decisions per episode before truncation (continuing task). 0 = never truncate here (rely on Max Step).")]
        public int episodeDecisionLimit = 2048;
        public int seed = 1;

        [Header("Optional read-only view")]
        public ElevatorRenderer view;

        public Building Sim => _b;

        Building _b;
        float _clock;
        int _decisions;

        public override void Initialize()
        {
            _b = new Building(buildingConfig, rewardConfig, observationConfig, trafficConfig, seed);

            var bp = GetComponent<BehaviorParameters>();
            var branches = new int[buildingConfig.numElevators];
            for (int i = 0; i < branches.Length; i++) branches[i] = 6;
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

            _b.Reset(forced);
            _clock = 0f;
            _decisions = 0;
        }

        void FixedUpdate()
        {
            if (_b == null) return;
            float dt = Time.fixedDeltaTime;
            _b.Tick(dt);

            _clock += dt;
            if (_clock >= buildingConfig.decisionInterval)
            {
                _clock -= buildingConfig.decisionInterval;
                _b.StepServiceChanges();

                // Skip the decision entirely if every car is mid-travel/door-cycle (all 5
                // non-NOOP actions masked for every car this tick) — with floorTravelTime
                // (1.6s default) several times decisionInterval (0.5s), most fixed-cadence
                // ticks would otherwise produce a fully-masked, wasted OnActionReceived call.
                // NOTE: this only removes wasted decisions on ticks with nothing to choose;
                // it does not remove the up-to-decisionInterval latency between a car actually
                // becoming idle and the policy next acting on it (that would need fully
                // event-driven decisions, a bigger change — see EXPERIMENT_PLAN.md action-space
                // notes). It also means episodeDecisionLimit now corresponds to a fleet/traffic
                // -dependent amount of sim time, since real decisions fire less often when fewer
                // cars are actionable.
                if (AnyCarNeedsDecision()) RequestDecision();
            }

            if (view != null) view.Mirror(_b);
        }

        bool AnyCarNeedsDecision()
        {
            for (int i = 0; i < _b.cars.Length; i++)
            {
                var c = _b.cars[i];
                if (c.inService && c.AtFloor) return true;
            }
            return false;
        }

        public override void CollectObservations(VectorSensor sensor) => _b.WriteObservation(sensor);

        public override void OnActionReceived(ActionBuffers actions)
        {
            var d = actions.DiscreteActions;
            for (int i = 0; i < buildingConfig.numElevators; i++)
                _b.ApplyAction(i, d[i]);

            AddReward(_b.CollectReward());

            _decisions++;
            if (episodeDecisionLimit > 0 && _decisions >= episodeDecisionLimit) EndEpisode();
        }

        public override void WriteDiscreteActionMask(IDiscreteActionMask mask)
        {
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
            var plan = ElevatorHeuristics.CollectiveLook(_b);
            for (int i = 0; i < plan.Length; i++) d[i] = plan[i];
        }
    }
}
