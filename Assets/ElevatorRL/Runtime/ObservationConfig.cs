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
        [Tooltip("Normalized age (waitTime/maxWait, clamped) of the LONGEST-waiting rider per " +
            "hall queue, up/down per floor. Without this the policy can see THAT a floor has a " +
            "call but not how close it is to abandoning (-8 reward) — real ETA controllers track " +
            "this. Real hall buttons don't report exact wait either, but this is the one piece of " +
            "\"realistic\" info a controller plausibly could infer/log, so it's grouped with the " +
            "other limited-observability blocks above rather than under full observability.")]
        public bool hallCallAge = true;  // 2F  (oldest wait per queue, up/down)

        [Header("Continuous-time additions (recommended)")]
        public bool carMotion = true;    // E x (dir one-hot[3] + normalized position)
        public bool carLoads = true;     // E   (riders / capacity)

        [Header("Full observability (ablation) & context")]
        public bool queueLengths = false; // 2F  (normalized by maxQueue)
        public bool timeOfDay = false;    // 2   (sin, cos of day phase)
        public bool pattern = false;      // 5   (one-hot traffic regime)
    }
}
