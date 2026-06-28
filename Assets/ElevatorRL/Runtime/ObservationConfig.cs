using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Flags that compose the vector observation. Toggle blocks to build the
    /// "limited" (realistic) space or a fully-observable ablation. WriteObservation
    /// appends blocks in a FIXED order; the policy's Space Size must equal
    /// Building.ObservationSize(). carActive should stay ON whenever the fleet size
    /// is variable so the policy knows which slots are live.
    /// </summary>
    [CreateAssetMenu(menuName = "Elevator/Observation Config", fileName = "ObservationConfig")]
    public sealed class ObservationConfig : ScriptableObject
    {
        [Header("Limited / partially observable (realistic)")]
        public bool carFloor = true;     // E x one-hot(F)
        public bool carActive = true;    // E   (1 = in service / present)
        public bool carButtons = true;   // E x F  (destination buttons pressed)
        public bool hallButtons = true;  // 2F  (up/down hall lamps)

        [Header("Continuous-time additions (recommended)")]
        public bool carMotion = true;    // E x (dir one-hot[3] + normalized position)
        public bool carLoads = true;     // E   (riders / capacity)

        [Header("Full observability (ablation) & context")]
        public bool queueLengths = false; // 2F  (normalized by maxQueue)
        public bool timeOfDay = false;    // 2   (sin, cos of day phase)
        public bool pattern = false;      // 5   (one-hot traffic regime)
    }
}
