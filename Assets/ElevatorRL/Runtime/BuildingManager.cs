using UnityEngine;
using Unity.MLAgents;

namespace ElevatorRL
{
    /// <summary>
    /// Owns the single shared Building for a multi-agent (per-car) training scene (EXPERIMENT_PLAN.md
    /// E6). Ticks the simulation every FixedUpdate, drives the shared decision cadence, and
    /// broadcasts one team reward to every ElevatorCarAgent each decision — this is a fully
    /// cooperative dispatch problem, so every car is credited for the whole building's outcome
    /// each step, not just its own action (see E6's reward-shaping note). Individual
    /// ElevatorCarAgents never touch Building.Reset()/Tick() themselves; this manager is the single
    /// source of truth for the sim clock and episode boundary so N agents sharing one Building
    /// don't race resetting or double-ticking it.
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
        Building _b;
        float _clock;
        int _decisions;

        void Awake()
        {
            _b = new Building(buildingConfig, rewardConfig, observationConfig, trafficConfig, seed);
            _carAgents = GetComponentsInChildren<ElevatorCarAgent>();
            for (int i = 0; i < _carAgents.Length; i++)
            {
                _carAgents[i].manager = this;
                _carAgents[i].carIndex = i;
            }
        }

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

            float reward = _b.CollectReward();
            for (int i = 0; i < _carAgents.Length; i++)
            {
                _carAgents[i].AddReward(reward);
                if (_carAgents[i].IsEligible()) _carAgents[i].RequestDecision();
            }

            _decisions++;
            if (episodeDecisionLimit > 0 && _decisions >= episodeDecisionLimit)
            {
                // EpisodeInterrupted (truncation, not termination) for the same reason the
                // single-agent design used it: this is a continuing task with no genuine terminal
                // state, so bootstrapping the value estimate at the cutoff is correct — EndEpisode()
                // would incorrectly tell PPO the final state's value is exactly 0.
                for (int i = 0; i < _carAgents.Length; i++) _carAgents[i].EpisodeInterrupted();
                ResetEpisode();
            }
        }
    }
}
