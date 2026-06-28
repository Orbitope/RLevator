using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Reward weights, ported verbatim from the reference environment. Positive terms
    /// are progress; negative terms are congestion / failure. The two OCCUPANCY terms
    /// (inElevator, inQueue) are multiplied by elapsed time so the trade-off is
    /// independent of decision cadence; event terms are counted as they happen.
    /// </summary>
    [CreateAssetMenu(menuName = "Elevator/Reward Config", fileName = "RewardConfig")]
    public sealed class RewardConfig : ScriptableObject
    {
        [Header("Positive")]
        public float delivered = 10.0f;     // per passenger unloaded at destination
        public float movedToward = 0.4f;    // per rider-floor travelled toward dest

        [Header("Negative (keep these <= 0)")]
        public float movedAway = -0.4f;     // per rider-floor travelled away from dest
        public float rejected = -5.0f;      // per arrival denied by a full queue
        public float abandoned = -8.0f;     // per queued rider past maxWait
        public float inElevator = -0.04f;   // per rider-second in transit
        public float inQueue = -0.12f;      // per passenger-second waiting
    }
}
