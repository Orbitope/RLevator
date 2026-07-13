using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

namespace ElevatorRL
{
    /// <summary>
    /// One instance per car (EXPERIMENT_PLAN.md E6, multi-agent/parameter-sharing architecture).
    /// Every car instance for a rung shares the same Behavior Name, so ML-Agents trains ONE policy
    /// shared across all of them — this is how "shared per-car encoder" is achieved without any
    /// custom network code. Owns no simulation state itself: BuildingManager owns the single
    /// shared Building, ticks it, and drives this agent's RequestDecision()/AddReward() calls; this
    /// component only translates that shared Building into this car's own local
    /// observation/action/mask (see Building.WriteCarObservation — same shape regardless of fleet
    /// size, which is what lets one policy generalize across car counts/rungs).
    /// </summary>
    [RequireComponent(typeof(BehaviorParameters))]
    public sealed class ElevatorCarAgent : Agent
    {
        // Assigned by BuildingManager.Awake(), not the inspector — see its GetComponentsInChildren.
        [System.NonSerialized] public BuildingManager manager;
        [System.NonSerialized] public int carIndex;

        public bool IsEligible()
        {
            var c = manager.Sim.cars[carIndex];
            return c.inService && c.AtFloor;
        }

        public override void CollectObservations(VectorSensor sensor) =>
            manager.Sim.WriteCarObservation(sensor, carIndex);

        // Single size-6 branch (noop/up/down/board-up/board-down/unload) for THIS car only — no
        // "mid-travel, force NOOP" case needed like the single-agent design had, since
        // BuildingManager simply never calls RequestDecision() for a car that isn't idle this tick.
        public override void WriteDiscreteActionMask(IDiscreteActionMask mask)
        {
            var b = manager.Sim;
            var c = b.cars[carIndex];

            if (c.Floor >= c.maxFloor) mask.SetActionEnabled(0, 1, false);
            if (c.Floor <= c.minFloor) mask.SetActionEnabled(0, 2, false);
            if (b.upQ[c.Floor].Count == 0) mask.SetActionEnabled(0, 3, false);
            if (b.downQ[c.Floor].Count == 0) mask.SetActionEnabled(0, 4, false);
            if (!c.WantsFloor(c.Floor)) mask.SetActionEnabled(0, 5, false);
        }

        public override void OnActionReceived(ActionBuffers actions) =>
            manager.Sim.ApplyAction(carIndex, actions.DiscreteActions[0]);

        // Debug/manual-testing only — always NOOP; not exercised by training (PPO doesn't call
        // Heuristic) or by the headless EvalHarness (which uses PpoDispatcher/heuristics directly).
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var d = actionsOut.DiscreteActions;
            d[0] = 0;
        }

        // BuildingManager owns Building.Reset() and synchronizes all car agents' episode
        // boundaries together (see its FixedUpdate) — an individual car agent resetting the shared
        // Building on its own OnEpisodeBegin would race with the other N-1 agents' resets.
        public override void OnEpisodeBegin() { }
    }
}
