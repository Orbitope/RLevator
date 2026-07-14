using UnityEngine;
using Unity.MLAgents;

namespace ElevatorRL
{
    /// <summary>
    /// Owns the single shared Building for a multi-agent (per-car) training scene (EXPERIMENT_PLAN.md
    /// E6, Architecture A). Ticks the simulation every FixedUpdate, drives the shared decision
    /// cadence, and credits the whole building's reward to a MA-POCA agent GROUP each decision.
    ///
    /// Uses SimpleMultiAgentGroup + AddGroupReward (trainer_type: poca), NOT plain per-agent
    /// AddReward + ppo. Elevator dispatch is a fully COOPERATIVE task, and independent PPO agents
    /// each treating a broadcast team reward as an individual reward — with a decentralized critic
    /// that sees only one car — failed to learn at all (flat reward, delivered=0; see the E6-A
    /// "independent-PPO" note in EXPERIMENT_PLAN.md). MA-POCA gives the group a CENTRALIZED critic
    /// over all cars, which is the right tool for this credit-assignment problem. Individual
    /// ElevatorCarAgents never touch Building.Reset()/Tick(); this manager is the single source of
    /// truth for the sim clock and the group episode boundary.
    /// </summary>
    public sealed class BuildingManager : MonoBehaviour
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

        public Building Sim => _b;
        ElevatorCarAgent[] _carAgents;
        SimpleMultiAgentGroup _group;
        Building _b;
        float _clock;
        int _decisions;

        void Awake()
        {
            _b = new Building(buildingConfig, rewardConfig, observationConfig, trafficConfig, seed);
            _carAgents = GetComponentsInChildren<ElevatorCarAgent>();
            _group = new SimpleMultiAgentGroup();
            for (int i = 0; i < _carAgents.Length; i++)
            {
                _carAgents[i].manager = this;
                _carAgents[i].carIndex = i;
                _group.RegisterAgent(_carAgents[i]);
            }
        }

        void OnDestroy() => _group?.Dispose();

        void Start() => ResetEpisode();

        void ResetEpisode()
        {
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
            if (_clock < buildingConfig.decisionInterval) return;
            _clock -= buildingConfig.decisionInterval;
            _b.StepServiceChanges();

            // Cooperative team reward to the whole group (MA-POCA centralized critic).
            // MA-POCA requires both group reward (for the critic) and per-agent rewards.
            // Distribute the shared building reward equally to all agents.
            float r = _b.CollectReward();
            _group.AddGroupReward(r);
            float perAgentReward = r / _carAgents.Length;
            for (int i = 0; i < _carAgents.Length; i++)
            {
                _carAgents[i].AddReward(perAgentReward);
                if (_carAgents[i].IsEligible()) _carAgents[i].RequestDecision();
            }

            _decisions++;
            if (episodeDecisionLimit > 0 && _decisions >= episodeDecisionLimit)
            {
                // GroupEpisodeInterrupted = truncation (not termination) for the whole group: this is
                // a continuing task with no genuine terminal state, so bootstrap the value estimate
                // at the cutoff rather than telling the trainer the final state's value is 0.
                _group.GroupEpisodeInterrupted();
                ResetEpisode();
            }
        }
    }
}
