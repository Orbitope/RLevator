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

        [Tooltip("2 x F: per-floor arrival RATES — [from-rate, to-rate] (arrivals/sec originating at, " +
            "and destined for, each floor). This is the 2024 Traffic-Pattern-Aware paper's actual " +
            "traffic-pattern-awareness mechanism (its 2 `rate` channels on a separate FC pathway), " +
            "which we were missing entirely — it is far richer than the `pattern` one-hot: it says " +
            "WHERE demand is and HOW MUCH, not just which named regime. Realistic (buildings know / " +
            "can measure their traffic profile; the paper blends nominal with a rolling 5-min measured " +
            "rate). NOTE this also encodes the LOAD LEVEL, so a rate-aware policy can in principle " +
            "adapt across intensities — directly relevant to the E13c train/eval regime mismatch. " +
            "See EXPERIMENT_PLAN.md E13e.")]
        public bool arrivalRates = false; // 2F  (from-rate, to-rate per floor)

        [Tooltip("EXACT destination histogram for every waiting AND in-car rider. The HALL portion " +
            "(2 x F x F) is a faithful model of a Destination Control System (DCS: Schindler " +
            "Miconic/PORT, Otis Compass, KONE, TK) where riders enter their destination at a lobby " +
            "kiosk/keycard BEFORE boarding -- so pre-boarding destinations ARE a real, deployed " +
            "signal, most valuable in interfloor-heavy buildings. The in-car E x F portion goes " +
            "beyond DCS (a real car knows its own selected destinations but not un-selected riders' " +
            "intent) and is the ceiling/ablation part. See EXPERIMENT_PLAN.md E12: the DCS-realistic " +
            "test drops the in-car block and keeps only the hall destinations. " +
            "2 x F x F (hall up/down, per origin floor, per destination floor) + E x F (per car, " +
            "per destination floor).")]
        public bool omniscientDestinations = false;
    }
}
