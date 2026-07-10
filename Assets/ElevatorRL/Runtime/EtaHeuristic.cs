using System.Collections.Generic;
using UnityEngine;

namespace ElevatorRL
{
    /// <summary>
    /// ETA/ETD (estimated-time-of-arrival) dispatch — the honest strong baseline, modeled on
    /// how real destination-dispatch group controllers assign hall calls: each waiting call is
    /// given to whichever in-service car has the lowest ESTIMATED TIME TO ARRIVE, given that
    /// car's current onboard passengers and its other already-committed calls — not merely
    /// whichever car is nearest by floor count.
    ///
    /// This is deliberately a different mechanism from <see cref="ElevatorHeuristics.CollectiveLook"/>:
    /// LOOK's dispatch pass only reassigns idle, EMPTY cars (for an empty car, nearest-floor
    /// distance already equals ETA under constant travel speed, so there is nothing to improve
    /// there). ETA/ETD instead allows ANY in-service car — including ones currently carrying
    /// riders — to pick up a new call if its estimated arrival beats every other car's. That is
    /// the property expected to matter increasingly at scale and under per-car floor
    /// restrictions (EXPERIMENT_PLAN.md §0/§1): a busy car already sweeping past a call may serve
    /// it far sooner than the nearest literally-idle car, and a floor-restricted car should never
    /// be shut out of a call within its own zone by a geometrically-closer car that can't legally
    /// serve that floor at all.
    ///
    /// STICKY BY DESIGN: a car only moves one floor per decision (see Building.ApplyAction — cases
    /// 1/2 step a single floor and the car re-idles), so the full dispatch recomputes every
    /// decision. Real ETA/ETD systems lock a call to its assigned car once committed, precisely to
    /// avoid oscillation; without that, a car nominally heading toward a call could be pulled onto
    /// a marginally-better-looking one every decision, "orphaning" the first (this was observed
    /// empirically as elevated lobby abandonment before stickiness was added). This class is
    /// therefore stateful — construct ONE instance per episode/Building and reuse it every
    /// decision — and remembers each car's assigned calls across decisions, releasing a call only
    /// when it is fully served or its owner leaves service.
    ///
    /// Shares only the Building/Elevator API with CollectiveLook, not its code — kept fully
    /// independent so each dispatcher is separately auditable and CollectiveLook (already
    /// validated against M0 instrumentation) is never at risk of an incidental behavior change.
    /// </summary>
    public sealed class EtaHeuristic
    {
        readonly List<int>[] _assigned; // per car: floors this car currently owns a hall-call claim on
        readonly bool _weightByQueueDepth;

        /// <param name="weightByQueueDepth">
        /// Pure ETA (false, default) assigns each call to its lowest-ETA car independently of how
        /// many people are waiting there — a documented real limitation: under directional traffic
        /// (e.g. up-peak) it can systematically under-serve the highest-volume origin (the lobby)
        /// in favor of nearer but sparser calls, since giving busy cars extra stops lengthens their
        /// round trip and slows their return to the floor with the most demand (see class remarks;
        /// this was observed empirically as elevated lobby abandonment despite similar aggregate
        /// wait). Setting this true processes calls in descending queue-size order first, so the
        /// busiest call claims the best available car before smaller calls compete for it — the
        /// standard fix (load-aware dispatch) and the stronger of the two ETA/ETD baselines.
        /// </param>
        public EtaHeuristic(int numElevators, bool weightByQueueDepth = false)
        {
            _assigned = new List<int>[numElevators];
            for (int i = 0; i < numElevators; i++) _assigned[i] = new List<int>();
            _weightByQueueDepth = weightByQueueDepth;
        }

        public int[] Dispatch(Building b)
        {
            int E = b.cfg.numElevators;
            var act = new int[E];
            var target = new int[E];
            var hasTarget = new bool[E];

            float stopPenalty = b.cfg.doorTime * 2f + b.cfg.dwellTime;
            float travelTime = b.cfg.floorTravelTime;

            // 1. prune claims that are resolved (queue emptied — served or abandoned) or whose
            //    owner left service; release them so they can be picked up fresh below.
            var claimed = new HashSet<int>();
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                var list = _assigned[i];
                if (!el.inService) { list.Clear(); continue; }

                for (int k = list.Count - 1; k >= 0; k--)
                {
                    int f = list[k];
                    bool stillWaiting = b.upQ[f].Count > 0 || b.downQ[f].Count > 0;
                    if (!stillWaiting) list.RemoveAt(k);
                }
                for (int k = 0; k < list.Count; k++) claimed.Add(list[k]);
            }

            // 2. seed committed-stop sets: onboard riders' distinct destinations UNION this car's
            //    sticky hall-call claims (both are "things this car must still stop for").
            var committed = new List<int>[E];
            for (int i = 0; i < E; i++)
            {
                var set = new HashSet<int>();
                var el = b.cars[i];
                for (int r = 0; r < el.riders.Count; r++) set.Add(el.riders[r].dest);
                for (int k = 0; k < _assigned[i].Count; k++) set.Add(_assigned[i][k]);
                committed[i] = new List<int>(set);
            }

            // 3. assign every NEWLY waiting hall call (not already claimed by a sticky owner) to
            //    the in-service, in-range, spare-capacity car with the lowest estimated ETA.
            //    Processing order: ascending floor index by default; if weighting by queue depth,
            //    largest queue first, so the busiest call claims the best car before smaller ones
            //    compete for it (see constructor remarks).
            var callFloors = new List<int>(b.cfg.numFloors);
            for (int f = 0; f < b.cfg.numFloors; f++)
            {
                if (claimed.Contains(f)) continue;
                if (b.upQ[f].Count > 0 || b.downQ[f].Count > 0) callFloors.Add(f);
            }
            if (_weightByQueueDepth)
            {
                // stable descending sort by queue size (insertion sort — small N, keeps ties in
                // ascending-floor order, matching the unweighted default's tiebreak).
                for (int a = 1; a < callFloors.Count; a++)
                {
                    int fv = callFloors[a];
                    int wv = b.upQ[fv].Count + b.downQ[fv].Count;
                    int j = a - 1;
                    while (j >= 0 && (b.upQ[callFloors[j]].Count + b.downQ[callFloors[j]].Count) < wv)
                    { callFloors[j + 1] = callFloors[j]; j--; }
                    callFloors[j + 1] = fv;
                }
            }

            for (int c = 0; c < callFloors.Count; c++)
            {
                int f = callFloors[c];
                int best = -1; float bestEta = float.MaxValue;
                for (int i = 0; i < E; i++)
                {
                    var el = b.cars[i];
                    if (!el.inService || el.Free <= 0) continue;
                    if (f < el.minFloor || f > el.maxFloor) continue;

                    float eta = Eta(el, committed[i], f, travelTime, stopPenalty);
                    if (eta < bestEta) { bestEta = eta; best = i; }
                }
                if (best >= 0) { committed[best].Add(f); _assigned[best].Add(f); }
            }

            // 4. derive each car's next target from its committed set: nearest stop ahead in its
            //    current direction, else nearest behind.
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                int bestF = -1, bestDist = int.MaxValue; bool foundAhead = false;
                for (int k = 0; k < committed[i].Count; k++)
                {
                    int s = committed[i][k];
                    bool ahead = el.dir > 0 ? s > el.Floor : s < el.Floor;
                    int dist = Mathf.Abs(s - el.Floor);
                    if (ahead)
                    {
                        if (!foundAhead || dist < bestDist) { bestF = s; bestDist = dist; foundAhead = true; }
                    }
                    else if (!foundAhead && dist < bestDist) { bestF = s; bestDist = dist; }
                }
                target[i] = bestF; hasTarget[i] = bestF >= 0;
            }

            // 5. resolve each car's target into a primitive action — identical shape to
            //    CollectiveLook's pass 3 (unload / opportunistic board / step / idle).
            for (int i = 0; i < E; i++)
            {
                var el = b.cars[i];
                if (!el.inService || !el.AtFloor) { act[i] = 0; continue; }

                int f = el.Floor;
                if (el.WantsFloor(f)) { act[i] = 5; continue; }

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

        /// <summary>
        /// One-reversal scan-time estimate: if f is ahead of the car in its current direction,
        /// ETA is direct travel time plus a stop penalty for every already-committed stop
        /// strictly between the car and f. Otherwise the car must finish its current sweep to
        /// its farthest committed stop, reverse, and travel back to f — paying the forward
        /// sweep's stops, the reversal, and any committed stops between the car and f on the
        /// way back.
        /// </summary>
        static float Eta(Elevator el, List<int> committed, int f, float travelTime, float stopPenalty)
        {
            int pos = el.Floor;
            if (committed.Count == 0)
                return travelTime * Mathf.Abs(f - pos); // idle & empty: pure travel, no stops

            int dir = el.dir;
            bool fAhead = dir > 0 ? f > pos : f < pos;

            if (fAhead)
            {
                int between = 0;
                for (int k = 0; k < committed.Count; k++)
                {
                    int s = committed[k];
                    bool onTheWay = dir > 0 ? (s > pos && s < f) : (s < pos && s > f);
                    if (onTheWay) between++;
                }
                return travelTime * Mathf.Abs(f - pos) + stopPenalty * between;
            }

            // reversal required: sweep to the farthest stop ahead (in dir), then back to f.
            int farthest = pos;
            int aheadCount = 0;
            for (int k = 0; k < committed.Count; k++)
            {
                int s = committed[k];
                bool ahead = dir > 0 ? s > pos : s < pos;
                if (!ahead) continue;
                aheadCount++;
                if (dir > 0 ? s > farthest : s < farthest) farthest = s;
            }

            int backStops = 0;
            for (int k = 0; k < committed.Count; k++)
            {
                int s = committed[k];
                if (s == f) continue;
                bool betweenPosAndF = f > pos ? (s > pos && s < f) : (s < pos && s > f);
                if (betweenPosAndF) backStops++;
            }

            return travelTime * (Mathf.Abs(farthest - pos) + Mathf.Abs(farthest - f))
                 + stopPenalty * (aheadCount + backStops);
        }
    }
}
