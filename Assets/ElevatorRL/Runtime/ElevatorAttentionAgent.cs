using UnityEngine;
using UnityEngine.Rendering;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace ElevatorRL
{
    /// <summary>
    /// E6 Architecture B (EXPERIMENT_PLAN.md E6 / plan peaceful-giggling-fern.md): a single agent
    /// with a MultiDiscrete E-branch joint action (like the flat-MLP ElevatorControllerAgent), but
    /// whose observation is split so ML-Agents applies a SHARED per-car encoder + residual
    /// self-attention over the cars. Global/building state goes to the VectorSensor
    /// (Building.WriteGlobalObservation); each car is one fixed-size entity appended to a
    /// BufferSensorComponent (Building.WriteCarEntity). The attention is entirely built into
    /// ML-Agents (EntityEmbedding/ResidualSelfAttention) — no custom Python — and exports to ONNX
    /// normally. This is the "shared-encoder network" arm; it keeps joint action selection (unlike
    /// the multi-agent Architecture A) while gaining permutation-invariant per-car encoding.
    ///
    /// Requires a BehaviorParameters + a BufferSensorComponent (ObservableSize = CarEntitySize,
    /// MaxNumObservables = numElevators) on the same GameObject. VectorObservationSize/ActionSpec
    /// are baked at editor time by MultiAgentSetup (like ElevatorSetup does for the flat agent).
    /// Primitive action space only (no AS1 here).
    /// </summary>
    [RequireComponent(typeof(BehaviorParameters))]
    [RequireComponent(typeof(BufferSensorComponent))]
    public sealed class ElevatorAttentionAgent : Agent
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
        BufferSensorComponent _bufferSensor;
        float[] _entityBuf;
        float _clock;
        int _decisions;

        public override void Initialize()
        {
            _b = new Building(buildingConfig, rewardConfig, observationConfig, trafficConfig, seed);
            _bufferSensor = GetComponent<BufferSensorComponent>();
            _entityBuf = new float[_b.CarEntitySize()];
        }

        public override void OnEpisodeBegin()
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
            if (_clock >= buildingConfig.decisionInterval)
            {
                _clock -= buildingConfig.decisionInterval;
                _b.StepServiceChanges();
                if (AnyCarAtFloor()) RequestDecision();
            }

            if (view != null && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
                view.Mirror(_b);
        }

        bool AnyCarAtFloor()
        {
            for (int i = 0; i < _b.cars.Length; i++)
                if (_b.cars[i].inService && _b.cars[i].AtFloor) return true;
            return false;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            _b.WriteGlobalObservation(sensor);
            // One entity per IN-SERVICE car; out-of-service cars are simply not appended, so the
            // attention naturally sees only the active fleet (BufferSensor pads/masks the rest).
            for (int i = 0; i < _b.cars.Length; i++)
            {
                if (!_b.cars[i].inService) continue;
                System.Array.Clear(_entityBuf, 0, _entityBuf.Length);
                _b.WriteCarEntity(_entityBuf, i);
                _bufferSensor.AppendObservation(_entityBuf);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var d = actions.DiscreteActions;
            for (int i = 0; i < buildingConfig.numElevators; i++)
                _b.ApplyAction(i, d[i]);

            AddReward(_b.CollectReward());

            _decisions++;
            // Truncation, not termination (continuing task) — same reasoning as the other agents.
            if (episodeDecisionLimit > 0 && _decisions >= episodeDecisionLimit) EpisodeInterrupted();
        }

        public override void WriteDiscreteActionMask(IDiscreteActionMask mask)
        {
            for (int i = 0; i < buildingConfig.numElevators; i++)
            {
                var c = _b.cars[i];
                if (!c.inService || !c.AtFloor)
                {
                    for (int a = 1; a <= 5; a++) mask.SetActionEnabled(i, a, false);
                    continue;
                }
                if (c.Floor >= c.maxFloor) mask.SetActionEnabled(i, 1, false);
                if (c.Floor <= c.minFloor) mask.SetActionEnabled(i, 2, false);
                if (_b.upQ[c.Floor].Count == 0) mask.SetActionEnabled(i, 3, false);
                if (_b.downQ[c.Floor].Count == 0) mask.SetActionEnabled(i, 4, false);
                if (!c.WantsFloor(c.Floor)) mask.SetActionEnabled(i, 5, false);
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
