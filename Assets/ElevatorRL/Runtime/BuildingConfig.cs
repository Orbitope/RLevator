using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Building topology and continuous-time constants. numElevators is the FIXED
    /// policy width (the trained maximum, e.g. 10). The number actually in service
    /// per episode is randomized between minActiveElevators and numElevators so one
    /// policy generalizes across 2..N cars and supports out-of-service cars.
    /// </summary>
    [CreateAssetMenu(menuName = "Elevator/Building Config", fileName = "BuildingConfig")]
    public sealed class BuildingConfig : ScriptableObject
    {
        [Header("Topology")]
        [Min(2)] public int numFloors = 8;
        [Tooltip("FIXED policy width = the maximum number of cars the network supports. Keep constant across training and inference.")]
        [Min(1)] public int numElevators = 10;
        [Min(1)] public int capacity = 8;
        [Min(1)] public int maxQueue = 12;

        [Header("Variable fleet / out-of-service")]
        [Tooltip("Randomize how many cars are in service each episode (domain randomization).")]
        public bool randomizeActive = true;
        [Min(1)] public int minActiveElevators = 2;
        [Tooltip("Per-decision probability of taking a car out of / back into service mid-episode. 0 disables.")]
        [Range(0f, 0.2f)] public float serviceChangeProbability = 0.0f;

        [Header("Continuous timing (seconds)")]
        [Min(0.05f)] public float floorTravelTime = 1.6f;   // time to traverse one floor
        [Min(0.05f)] public float doorTime = 0.8f;          // doors open OR close
        [Min(0.05f)] public float dwellTime = 1.2f;         // load / unload hold
        [Min(0.05f)] public float decisionInterval = 0.5f;  // agent decision cadence
        [Min(1f)] public float maxWait = 45f;               // queue abandon threshold

        [Header("Per-car service range (optional, length == numElevators)")]
        [Tooltip("min/max floor each car may visit. Leave empty for full-building service. Use to model low-rise / high-rise banks.")]
        public Vector2Int[] floorRange;

        public int MinFloor(int car)
        {
            if (floorRange != null && car < floorRange.Length) return Mathf.Clamp(floorRange[car].x, 0, numFloors - 1);
            return 0;
        }

        public int MaxFloor(int car)
        {
            if (floorRange != null && car < floorRange.Length) return Mathf.Clamp(floorRange[car].y, 0, numFloors - 1);
            return numFloors - 1;
        }
    }
}
