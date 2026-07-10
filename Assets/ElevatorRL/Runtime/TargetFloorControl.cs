using System.Collections.Generic;
using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Shared logic for the AS1 "target-floor" (semi-MDP / options) action space
    /// (EXPERIMENT_PLAN.md §7 / E9). In AS1 a decision is "commit car i to target floor f"; a
    /// fixed deterministic sub-controller then drives the car floor-by-floor toward f — servicing
    /// opportunistic same-direction hall calls and unloading en route — until it arrives and has
    /// nothing left to do at the target, at which point the policy is asked for a new target.
    ///
    /// This class holds the two reusable, stateless pieces:
    ///   • <see cref="ResolveTowardTarget"/> — the deterministic sub-controller (committed target →
    ///     one primitive action), used BOTH by <see cref="ElevatorControllerAgent"/> in AS1 mode
    ///     and by the headless <c>TargetFloorLook</c> validation dispatcher in EvalHarness.
    ///   • <see cref="CommitmentFulfilled"/> — whether a committed car has finished at its target
    ///     (so the agent knows when to re-query the policy).
    ///   • <see cref="LookTargets"/> — a LOOK-equivalent *target* selector (mirrors CollectiveLook
    ///     passes 1–2), used as the AS1 heuristic / demonstration policy and for headless
    ///     validation that the sub-controller reproduces LOOK-like behavior.
    ///
    /// The sub-controller emits the same primitive action codes as <see cref="Building.ApplyAction"/>
    /// (0 noop / 1 up / 2 down / 3 board-up / 4 board-down / 5 unload) and never produces an action
    /// that would be illegal under <see cref="ElevatorControllerAgent.WriteDiscreteActionMask"/>,
    /// provided the target is within the car's [minFloor, maxFloor] range.
    /// </summary>
    public static class TargetFloorControl
    {
        /// <summary>
        /// Deterministic sub-controller: given a car committed to <paramref name="target"/>, return
        /// the primitive action to take now. Services the current floor first (unload, then
        /// opportunistic same-direction boarding), otherwise steps one floor toward the target.
        /// Returns 0 (noop) when idle at the target with nothing to do — the signal that the
        /// commitment is fulfilled and the policy should choose a new target.
        /// </summary>
        public static int ResolveTowardTarget(Building b, int carIndex, int target)
        {
            var c = b.cars[carIndex];
            if (!c.inService || !c.AtFloor) return 0; // moving / door cycle / OOS: nothing to issue

            int f = c.Floor;

            // 1. drop anyone who wants off here (always, regardless of target)
            if (c.WantsFloor(f)) return 5;

            if (f == target)
            {
                // 2a. arrived: board whoever is waiting (prefer up, then down) if there's room,
                //     else the commitment is fulfilled.
                if (c.Free > 0)
                {
                    if (b.upQ[f].Count > 0) return 3;
                    if (b.downQ[f].Count > 0) return 4;
                }
                return 0;
            }

            // 2b. en route: opportunistic same-direction boarding, then step toward the target.
            int dir = target > f ? 1 : -1;
            if (c.Free > 0)
            {
                if (dir > 0 && b.upQ[f].Count > 0) return 3;
                if (dir < 0 && b.downQ[f].Count > 0) return 4;
            }
            return dir > 0 ? 1 : 2;
        }

        /// <summary>
        /// True when a committed car is idle at its target with nothing left to unload or board —
        /// i.e. <see cref="ResolveTowardTarget"/> would return 0. The agent uses this to decide
        /// when to release the commitment and re-query the policy.
        /// </summary>
        public static bool CommitmentFulfilled(Building b, int carIndex, int target)
        {
            var c = b.cars[carIndex];
            if (!c.inService) return true;      // OOS car: drop the commitment
            if (!c.AtFloor) return false;       // mid-travel / door cycle: still working
            int f = c.Floor;
            if (f != target) return false;      // not there yet
            if (c.WantsFloor(f)) return false;  // still unloading
            if (c.Free > 0 && (b.upQ[f].Count > 0 || b.downQ[f].Count > 0)) return false; // still boarding
            return true;
        }

        /// <summary>
        /// LOOK-equivalent target selection: the floor each car would head toward under collective
        /// LOOK + nearest-car dispatch. Mirrors <see cref="ElevatorHeuristics.CollectiveLook"/>
        /// passes 1–2 (rider LOOK, then empty-car nearest-call claim) but returns the chosen TARGET
        /// FLOOR per car (falling back to the car's current floor = "hold") instead of a primitive
        /// action. Used as the AS1 heuristic/demonstration policy and for headless validation.
        /// Kept structurally parallel to CollectiveLook so the two stay comparable; it deliberately
        /// does not share code, so CollectiveLook (validated against M0 instrumentation) is never at
        /// risk of an incidental change.
        /// </summary>
        public static int[] LookTargets(Building b)
        {
            int E = b.cfg.numElevators;
            var target = new int[E];
            var hasTarget = new bool[E];

            // pass 1 — occupied cars head to the nearest desired floor, preferring current direction
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!el.inService || el.riders.Count == 0) { hasTarget[i] = false; continue; }

                int best = -1, bestDist = int.MaxValue; bool foundAhead = false;
                for (int r = 0; r < el.riders.Count; r++)
                {
                    int d = el.riders[r].dest;
                    bool ahead = el.dir > 0 ? d > el.Floor : d < el.Floor;
                    int dist = Mathf.Abs(d - el.Floor);
                    if (ahead)
                    {
                        if (!foundAhead || dist < bestDist) { best = d; bestDist = dist; foundAhead = true; }
                    }
                    else if (!foundAhead && dist < bestDist) { best = d; bestDist = dist; }
                }
                target[i] = best; hasTarget[i] = best >= 0;
            }

            // pass 2 — empty idle cars claim the nearest unclaimed in-range hall call
            var claimed = new HashSet<int>();
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!el.inService || el.riders.Count > 0) continue;

                int best = -1, bd = int.MaxValue;
                for (int f = el.minFloor; f <= el.maxFloor; f++)
                {
                    if (b.upQ[f].Count == 0 && b.downQ[f].Count == 0) continue;
                    if (claimed.Contains(f)) continue;
                    int d = Mathf.Abs(f - el.Floor);
                    if (d < bd) { bd = d; best = f; }
                }
                if (best >= 0) { claimed.Add(best); target[i] = best; hasTarget[i] = true; }
            }

            // no target → hold at current floor (clamped into range for safety)
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!hasTarget[i]) target[i] = Mathf.Clamp(el.Floor, el.minFloor, el.maxFloor);
            }

            return target;
        }
    }
}
