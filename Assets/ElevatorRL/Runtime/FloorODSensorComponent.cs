using UnityEngine;
using Unity.MLAgents.Sensors;

namespace ElevatorRL
{
    /// <summary>
    /// EXPERIMENT_PLAN.md E13d. Drop next to an <see cref="ElevatorControllerAgent"/> to add the
    /// (2 x F x F) origin×destination conv pathway alongside the agent's flat VectorSensor.
    ///
    /// IMPORTANT: remove the E13b <see cref="FloorGridSensorComponent"/> when using this — (a) the
    /// experiment is cleaner isolating the OD grid, and (b) `vis_encode_type` applies to ALL visual
    /// observations, so an 8-wide floor grid present at the same time would break `resnet`
    /// (min-resolution 15). Menu: Tools/Elevator RL/E13 Conv/Add Floor-OD Sensor To Agent.
    ///
    /// Ordering note: Agent.LazyInitialize runs Initialize() (builds the Building) BEFORE
    /// InitializeSensors() (this CreateSensors()), so agent.Sim is live here.
    /// </summary>
    [RequireComponent(typeof(ElevatorControllerAgent))]
    public sealed class FloorODSensorComponent : SensorComponent
    {
        [Tooltip("Sensor name; must be stable across train/eval (maps to an obs input at inference). " +
            "Sorts before 'VectorSensor_size*', so this takes obs_0 and the flat vector takes obs_1.")]
        public string sensorName = "FloorOD";

        public override ISensor[] CreateSensors()
        {
            var agent = GetComponent<ElevatorControllerAgent>();
            if (agent == null || agent.Sim == null)
            {
                Debug.LogError("[ElevatorRL] FloorODSensorComponent: no ElevatorControllerAgent with " +
                    "an initialized Building on this GameObject.");
                return System.Array.Empty<ISensor>();
            }
            return new ISensor[] { new FloorODSensor(agent.Sim, sensorName) };
        }
    }
}
