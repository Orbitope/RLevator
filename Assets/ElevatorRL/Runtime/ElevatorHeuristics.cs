using System.Collections.Generic;
using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// Collective LOOK + nearest-car baseline, ported from the reference choose().
    /// Skips out-of-service cars and only commands cars that are idle at a floor.
    /// Use it as a sanity baseline, for demos, or to record demonstrations (BC/GAIL).
    /// </summary>
    public static class ElevatorHeuristics
    {
        public static int[] CollectiveLook(Building b)
        {
            int E = b.cfg.numElevators;
            var act = new int[E];
            var target = new int[E];
            var hasTarget = new bool[E];

            // 1. cars with riders head to the nearest desired floor (prefer current direction)
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!el.inService) { hasTarget[i] = false; continue; }
                if (el.riders.Count == 0) { hasTarget[i] = false; continue; }

                int best = -1; int bestDist = int.MaxValue; bool foundAhead = false;
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

            // 2. assign hall calls to nearest idle empty cars (greedy, no double-claim)
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

            // 3. resolve each car into an action
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!el.inService || !el.AtFloor) { act[i] = 0; continue; }

                int f = el.Floor;
                if (el.WantsFloor(f)) { act[i] = 5; continue; }     // unload

                int rem = el.Free;
                if (rem > 0)
                {
                    if (el.riders.Count == 0)
                    {
                        if (b.upQ[f].Count > 0 && (!hasTarget[i] || target[i] >= f)) { act[i] = 3; continue; }
                        if (b.downQ[f].Count > 0) { act[i] = 4; continue; }
                        if (b.upQ[f].Count > 0) { act[i] = 3; continue; }
                    }
                    else
                    {
                        if (el.dir > 0 && b.upQ[f].Count > 0) { act[i] = 3; continue; }
                        if (el.dir < 0 && b.downQ[f].Count > 0) { act[i] = 4; continue; }
                    }
                }

                if (!hasTarget[i]) { act[i] = 0; continue; }
                if (target[i] > f) { act[i] = 1; continue; }
                if (target[i] < f) { act[i] = 2; continue; }
                if (rem > 0 && b.upQ[f].Count > 0) { act[i] = 3; continue; }
                if (rem > 0 && b.downQ[f].Count > 0) { act[i] = 4; continue; }
                act[i] = 0;
            }

            return act;
        }
    }
}
