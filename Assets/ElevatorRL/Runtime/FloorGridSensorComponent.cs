using UnityEngine;
using Unity.MLAgents.Sensors;

namespace ElevatorRL
{
    /// <summary>
    /// EXPERIMENT_PLAN.md E13b. Drop this next to an <see cref="ElevatorControllerAgent"/> to add a
    /// floor-axis convolutional observation pathway (a <see cref="FloorGridSensor"/>) ON TOP OF the
    /// agent's existing flat VectorSensor -- no agent code change needed. ML-Agents concatenates the
    /// visual CNN encoder's output with the vector encoder's before the policy trunk, so the flat obs
    /// keeps carrying car/global state while the grid gives the network floor-adjacency inductive
    /// bias. Set network_settings.vis_encode_type: match3 in the training yaml (min resolution 5,
    /// satisfied by F>=5 floors x 8 features).
    ///
    /// Ordering note: Agent.LazyInitialize calls Initialize() (which builds the Building) BEFORE
    /// InitializeSensors() (which calls this CreateSensors()), so agent.Sim is live here.
    /// </summary>
    [RequireComponent(typeof(ElevatorControllerAgent))]
    public sealed class FloorGridSensorComponent : SensorComponent
    {
        [Tooltip("Sensor name; must be stable across train/eval (maps to an obs input at inference).")]
        public string sensorName = "FloorGrid";

        public override ISensor[] CreateSensors()
        {
            var agent = GetComponent<ElevatorControllerAgent>();
            if (agent == null || agent.Sim == null)
            {
                Debug.LogError("[ElevatorRL] FloorGridSensorComponent: no ElevatorControllerAgent " +
                    "with an initialized Building found on this GameObject.");
                return System.Array.Empty<ISensor>();
            }
            return new ISensor[] { new FloorGridSensor(agent.Sim, sensorName) };
        }
    }
}
