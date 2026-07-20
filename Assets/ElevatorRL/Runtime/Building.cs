using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents.Sensors;

namespace ElevatorRL
{
    /// <summary>
    /// The whole environment, free of MonoBehaviour so it can run headless at high
    /// time-scale. Tick(dt) advances physics every FixedUpdate; ApplyAction is called
    /// once per car at each decision; CollectReward drains the accumulated terms.
    ///
    /// Variable fleet: cars.Length == cfg.numElevators is the FIXED policy width. Each
    /// episode marks a random subset in service (or honours a forced count) and zeros
    /// the observation / masks the actions of out-of-service cars.
    /// </summary>
    public sealed class Building
    {
        struct Acc
        {
            public float delivered, rejected, abandoned, toward, away, riderSeconds, queueSeconds;
        }

        public readonly BuildingConfig cfg;
        public readonly RewardConfig reward;
        public readonly ObservationConfig obs;
        public readonly TrafficConfig traffic;

        public Elevator[] cars;
        public List<Passenger>[] upQ;
        public List<Passenger>[] downQ;
        public PassengerArrivals arrivals;

        public float simTime;
        public TrafficPattern ActivePattern { get; private set; }

        // lifetime metrics (for HUD / logging only)
        public int DeliveredTotal, RejectedTotal, AbandonedTotal;
        public float WaitSum; public int WaitCount; public float MaxWaitObserved;

        // lifetime reward decomposition (for logging / stats; accumulated in CollectReward)
        public double RwTotal, RwDelivered, RwToward, RwAway, RwRejected, RwAbandoned, RwInElevator, RwInQueue;

        // observational events for telemetry only (no effect on sim; null when unsubscribed)
        public event Action<Passenger> OnArrival;          // queued (not rejected)
        public event Action<Passenger, float> OnDelivered; // (passenger, rideSeconds)
        public event Action<Passenger> OnAbandoned;        // left a hall queue past maxWait
        public event Action<int> OnRejected;               // arrival denied at floor (full queue)

        Acc _acc;
        readonly System.Random _rng;

        public Building(BuildingConfig cfg, RewardConfig reward, ObservationConfig obs, TrafficConfig traffic, int seed)
        {
            this.cfg = cfg;
            this.reward = reward;
            this.obs = obs;
            this.traffic = traffic;
            _rng = new System.Random(seed);

            arrivals = new PassengerArrivals(cfg.numFloors, seed);
            cars = new Elevator[cfg.numElevators];
            for (int i = 0; i < cars.Length; i++)
                cars[i] = new Elevator(i, cfg.MinFloor(i), cfg.MaxFloor(i), cfg.capacity, cfg.MinFloor(i));

            upQ = new List<Passenger>[cfg.numFloors];
            downQ = new List<Passenger>[cfg.numFloors];
            for (int f = 0; f < cfg.numFloors; f++) { upQ[f] = new List<Passenger>(); downQ[f] = new List<Passenger>(); }
        }

        int _pid;

        /// <param name="forcedActive">-1 = randomize per config; otherwise force this many cars in service.</param>
        public void Reset(int forcedActive = -1)
        {
            simTime = 0f;
            _pid = 0;
            _acc = default;
            DeliveredTotal = RejectedTotal = AbandonedTotal = WaitCount = 0;
            WaitSum = MaxWaitObserved = 0f;
            RwTotal = RwDelivered = RwToward = RwAway = RwRejected = RwAbandoned = RwInElevator = RwInQueue = 0;

            for (int f = 0; f < cfg.numFloors; f++) { upQ[f].Clear(); downQ[f].Clear(); }

            // spread starting floors across each car's range
            for (int i = 0; i < cars.Length; i++)
            {
                int lo = cfg.MinFloor(i), hi = cfg.MaxFloor(i);
                int start = cars.Length > 1 ? lo + (int)Math.Round(i * (hi - lo) / (double)(cars.Length - 1)) : lo;
                cars[i].HardReset(Mathf.Clamp(start, lo, hi));
            }

            // choose how many cars are in service this episode
            int active;
            if (forcedActive > 0) active = Mathf.Clamp(forcedActive, 1, cfg.numElevators);
            else if (cfg.randomizeActive) active = _rng.Next(cfg.minActiveElevators, cfg.numElevators + 1);
            else active = cfg.numElevators;

            // turn off a random distinct subset of size (numElevators - active)
            int off = cfg.numElevators - active;
            int[] order = new int[cfg.numElevators];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            for (int i = order.Length - 1; i > 0; i--) { int j = _rng.Next(i + 1); (order[i], order[j]) = (order[j], order[i]); }
            for (int k = 0; k < off; k++) cars[order[k]].inService = false;

            UpdatePattern(true);
        }

        public int CountInService()
        {
            int n = 0;
            for (int i = 0; i < cars.Length; i++) if (cars[i].inService) n++;
            return n;
        }

        // ---------------------------------------------------------------- per-FixedUpdate tick
        public void Tick(float dt)
        {
            UpdatePattern(false);
            ExpireQueues();
            SpawnArrivals(dt);
            AdvanceCars(dt);
            AgeOccupants(dt);
            simTime += dt;
        }

        void UpdatePattern(bool force)
        {
            // E15: rates are now TIME-VARYING (48 x 15-min interpolated bins), so this must recompute
            // every tick, not just when the named pattern changes. Cost is 3 interpolations + F + F*F
            // writes — negligible. `useDayCycle` means the paper's "AllInOne": the full 12h day profile
            // rather than a 3h slice, so the day cycle now falls out of the rate tables themselves
            // instead of switching between hand-tuned pattern presets.
            bool allInOne = traffic.useDayCycle;
            TrafficPattern want = allInOne ? traffic.PatternForTime(simTime) : traffic.defaultPattern;
            ActivePattern = want;
            // V0: population derived from fleet carrying capacity so every rung sits at the paper's
            // ~loadPerSlot arrivals/slot with no per-rung asset juggling (S=360, M=600, L=960, H=1200).
            int population = cfg.numElevators * cfg.capacity * traffic.loadPerSlot;
            arrivals.LoadPatternAtTime(want, allInOne, simTime, population);
        }

        void SpawnArrivals(float dt)
        {
            for (int f = 0; f < cfg.numFloors; f++)
            {
                int n = arrivals.Poisson(arrivals.lambda[f] * traffic.intensity * dt);
                for (int k = 0; k < n; k++)
                {
                    int d = arrivals.SampleDest(f);
                    if (d == f) continue;
                    var p = new Passenger(_pid++, f, d, simTime);
                    var q = p.Dir > 0 ? upQ[f] : downQ[f];
                    if (q.Count >= cfg.maxQueue) { _acc.rejected++; RejectedTotal++; OnRejected?.Invoke(f); }
                    else { q.Add(p); OnArrival?.Invoke(p); }
                }
            }
        }

        void ExpireQueues()
        {
            for (int f = 0; f < cfg.numFloors; f++)
            {
                ExpireOne(upQ[f]);
                ExpireOne(downQ[f]);
            }
        }

        void ExpireOne(List<Passenger> q)
        {
            for (int i = q.Count - 1; i >= 0; i--)
                if (q[i].waitTime >= cfg.maxWait) { var pp = q[i]; q.RemoveAt(i); _acc.abandoned++; AbandonedTotal++; OnAbandoned?.Invoke(pp); }
        }

        void AdvanceCars(float dt)
        {
            float speed = 1f / Mathf.Max(0.0001f, cfg.floorTravelTime); // floors / second
            for (int i = 0; i < cars.Length; i++)
            {
                var c = cars[i];
                if (!c.inService) continue;

                switch (c.state)
                {
                    case CarState.Moving:
                    {
                        float old = c.position;
                        c.position = Mathf.MoveTowards(c.position, c.target, speed * dt);
                        float moved = c.position - old;
                        if (Mathf.Abs(moved) > 0f)
                        {
                            float am = Mathf.Abs(moved);
                            for (int r = 0; r < c.riders.Count; r++)
                            {
                                float toDest = c.riders[r].dest - old;
                                if (Mathf.Sign(toDest) == Mathf.Sign(moved)) _acc.toward += am;
                                else _acc.away += am;
                            }
                        }
                        if (Mathf.Abs(c.position - c.target) < 1e-4f) { c.position = c.target; c.state = CarState.Idle; }
                        break;
                    }
                    case CarState.DoorsOpening:
                        c.timer -= dt;
                        if (c.timer <= 0f) { DoTransfer(c); c.state = CarState.Dwelling; c.timer = cfg.dwellTime; }
                        break;
                    case CarState.Dwelling:
                        c.timer -= dt;
                        if (c.timer <= 0f) { c.state = CarState.DoorsClosing; c.timer = cfg.doorTime; }
                        break;
                    case CarState.DoorsClosing:
                        c.timer -= dt;
                        if (c.timer <= 0f) { c.state = CarState.Idle; c.pending = 0; }
                        break;
                }
            }
        }

        void DoTransfer(Elevator c)
        {
            int f = c.Floor;
            if (c.pending == 5) // unload
            {
                for (int i = c.riders.Count - 1; i >= 0; i--)
                {
                    var p = c.riders[i];
                    if (p.dest == f)
                    {
                        _acc.delivered++; DeliveredTotal++;
                        WaitSum += p.waitTime; WaitCount++;
                        OnDelivered?.Invoke(p, p.age - p.waitTime);
                        c.riders.RemoveAt(i);
                    }
                }
            }
            else if (c.pending == 3) // load up-queue
            {
                var q = upQ[f];
                while (c.Free > 0 && q.Count > 0) { c.riders.Add(q[0]); q.RemoveAt(0); }
                c.dir = 1;
            }
            else if (c.pending == 4) // load down-queue
            {
                var q = downQ[f];
                while (c.Free > 0 && q.Count > 0) { c.riders.Add(q[0]); q.RemoveAt(0); }
                c.dir = -1;
            }
        }

        void AgeOccupants(float dt)
        {
            int waiting = 0;
            for (int f = 0; f < cfg.numFloors; f++)
            {
                waiting += AgeQueue(upQ[f], dt);
                waiting += AgeQueue(downQ[f], dt);
            }
            _acc.queueSeconds += waiting * dt;

            int inCar = 0;
            for (int i = 0; i < cars.Length; i++)
            {
                var c = cars[i];
                for (int r = 0; r < c.riders.Count; r++) c.riders[r].age += dt;
                inCar += c.riders.Count;
            }
            _acc.riderSeconds += inCar * dt;
        }

        int AgeQueue(List<Passenger> q, float dt)
        {
            for (int i = 0; i < q.Count; i++)
            {
                q[i].waitTime += dt; q[i].age += dt;
                if (q[i].waitTime > MaxWaitObserved) MaxWaitObserved = q[i].waitTime;
            }
            return q.Count;
        }

        // ---------------------------------------------------------------- per-decision
        public void ApplyAction(int carIndex, int action)
        {
            var c = cars[carIndex];
            if (!c.inService || !c.AtFloor) return; // command ignored unless idle & in service

            switch (action)
            {
                case 1: if (c.Floor < c.maxFloor) { c.target = c.Floor + 1; c.dir = 1; c.state = CarState.Moving; } break;
                case 2: if (c.Floor > c.minFloor) { c.target = c.Floor - 1; c.dir = -1; c.state = CarState.Moving; } break;
                case 3: case 4: case 5:
                    c.pending = action; c.state = CarState.DoorsOpening; c.timer = cfg.doorTime; break;
                default: break; // 0 = NOOP
            }
        }

        public float CollectReward()
        {
            float rDel = reward.delivered   * _acc.delivered;
            float rTow = reward.movedToward * _acc.toward;
            float rAwy = reward.movedAway   * _acc.away;
            float rRej = reward.rejected    * _acc.rejected;
            float rAbn = reward.abandoned   * _acc.abandoned;
            float rInE = reward.inElevator  * _acc.riderSeconds;
            float rInQ = reward.inQueue     * _acc.queueSeconds;
            float r = rDel + rTow + rAwy + rRej + rAbn + rInE + rInQ;

            RwDelivered += rDel; RwToward += rTow; RwAway += rAwy; RwRejected += rRej;
            RwAbandoned += rAbn; RwInElevator += rInE; RwInQueue += rInQ; RwTotal += r;

            _acc = default;
            return r;
        }

        // ---------------------------------------------------------------- mid-episode service changes
        public void StepServiceChanges()
        {
            if (!cfg.randomizeActive || cfg.serviceChangeProbability <= 0f) return;
            if (_rng.NextDouble() >= cfg.serviceChangeProbability) return;

            int active = CountInService();
            bool canDown = active > cfg.minActiveElevators;
            bool canUp = active < cfg.numElevators;
            if (!canDown && !canUp) return;

            if (canDown && (!canUp || _rng.Next(2) == 0)) TakeOutOfService(PickInService());
            else BringIntoService(PickOutOfService());
        }

        void TakeOutOfService(int idx)
        {
            if (idx < 0) return;
            var c = cars[idx];
            int f = c.Floor;
            // evacuate riders at the current floor; non-arrived riders re-enter the hall queue
            for (int i = 0; i < c.riders.Count; i++)
            {
                var p = c.riders[i];
                if (p.dest == f) continue;
                p.origin = f;
                (p.Dir > 0 ? upQ[f] : downQ[f]).Add(p);
            }
            c.riders.Clear();
            c.inService = false;
            c.state = CarState.Idle;
            c.pending = 0; c.timer = 0f;
        }

        void BringIntoService(int idx)
        {
            if (idx < 0) return;
            cars[idx].inService = true;
            cars[idx].state = CarState.Idle;
        }

        int PickInService()
        {
            int count = 0; for (int i = 0; i < cars.Length; i++) if (cars[i].inService) count++;
            if (count == 0) return -1;
            int pick = _rng.Next(count);
            for (int i = 0; i < cars.Length; i++) if (cars[i].inService && pick-- == 0) return i;
            return -1;
        }

        int PickOutOfService()
        {
            int count = 0; for (int i = 0; i < cars.Length; i++) if (!cars[i].inService) count++;
            if (count == 0) return -1;
            int pick = _rng.Next(count);
            for (int i = 0; i < cars.Length; i++) if (!cars[i].inService && pick-- == 0) return i;
            return -1;
        }

        // ---------------------------------------------------------------- observation
        public int ObservationSize()
        {
            int E = cfg.numElevators, F = cfg.numFloors, s = 0;
            if (obs.carFloor) s += E * F;
            if (obs.carActive) s += E;
            if (obs.carButtons) s += E * F;
            if (obs.hallButtons) s += 2 * F;
            if (obs.hallCallAge) s += 2 * F;
            if (obs.carMotion) s += 4 * E;
            if (obs.carLoads) s += E;
            if (obs.queueLengths) s += 2 * F;
            if (obs.timeOfDay) s += 2;
            if (obs.pattern) s += 5;
            if (obs.arrivalRates) s += 2 * F;
            if (obs.omniscientDestinations) s += 2 * F * F + E * F;
            return s;
        }

        // EXPERIMENT_PLAN.md E13e. Per-floor arrival rates: [from-rate, to-rate] in arrivals/sec.
        //   from-rate[f] = lambda[f] * intensity                      (demand ORIGINATING at f)
        //   to-rate[f]   = sum_o lambda[o] * intensity * destDist[o][f]  (demand DESTINED for f)
        // This is the 2024 paper's traffic-pattern-awareness mechanism (its 2 rate channels), which we
        // lacked entirely. Richer than the `pattern` one-hot: it localizes demand rather than naming a
        // regime. Values are clamped to [0,1] (rates here are < ~1/s; UpPeak's lobby peaks near 0.9).
        // Uses the NOMINAL rates (the known traffic profile); the paper blends nominal with a rolling
        // 5-min MEASURED rate (`b*arm + (1-b)*real`) — a measured variant is a later refinement.
        public void FillArrivalRates(float[] buf)
        {
            int F = cfg.numFloors;
            float I = traffic != null ? traffic.intensity : 1f;
            for (int f = 0; f < F; f++)
            {
                buf[f] = Mathf.Min(1f, arrivals.lambda[f] * I);          // from-rate
                float to = 0f;
                for (int o = 0; o < F; o++)
                    if (o != f) to += arrivals.lambda[o] * I * arrivals.destDist[o][f];
                buf[F + f] = Mathf.Min(1f, to);                          // to-rate
            }
        }

        float[] _rateScratch;

        public void WriteObservation(VectorSensor sensor)
        {
            int E = cfg.numElevators, F = cfg.numFloors;

            if (obs.carFloor)
                for (int i = 0; i < E; i++)
                {
                    var c = cars[i];
                    if (c.inService) sensor.AddOneHotObservation(Mathf.Clamp(c.Floor, 0, F - 1), F);
                    else for (int k = 0; k < F; k++) sensor.AddObservation(0f);
                }

            if (obs.carActive)
                for (int i = 0; i < E; i++) sensor.AddObservation(cars[i].inService ? 1f : 0f);

            if (obs.carButtons)
                for (int i = 0; i < E; i++)
                {
                    var c = cars[i];
                    for (int f = 0; f < F; f++) sensor.AddObservation(c.inService && c.WantsFloor(f) ? 1f : 0f);
                }

            if (obs.hallButtons)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(upQ[f].Count > 0 ? 1f : 0f);
                    sensor.AddObservation(downQ[f].Count > 0 ? 1f : 0f);
                }

            if (obs.hallCallAge)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(OldestWaitFrac(upQ[f]));
                    sensor.AddObservation(OldestWaitFrac(downQ[f]));
                }

            if (obs.carMotion)
                for (int i = 0; i < E; i++)
                {
                    var c = cars[i];
                    if (c.inService)
                    {
                        int m = c.state == CarState.Moving ? (c.dir > 0 ? 2 : 0) : 1; // 0 down, 1 stopped, 2 up
                        sensor.AddOneHotObservation(m, 3);
                        sensor.AddObservation(F > 1 ? c.position / (F - 1) : 0f);
                    }
                    else { sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); }
                }

            if (obs.carLoads)
                for (int i = 0; i < E; i++) sensor.AddObservation(cars[i].inService ? cars[i].Load : 0f);

            if (obs.queueLengths)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(Mathf.Min(1f, upQ[f].Count / (float)cfg.maxQueue));
                    sensor.AddObservation(Mathf.Min(1f, downQ[f].Count / (float)cfg.maxQueue));
                }

            if (obs.timeOfDay)
            {
                float frac = traffic.DayFraction(simTime);
                sensor.AddObservation(Mathf.Sin(2f * Mathf.PI * frac));
                sensor.AddObservation(Mathf.Cos(2f * Mathf.PI * frac));
            }

            if (obs.pattern)
                sensor.AddOneHotObservation((int)ActivePattern, 5);

            // E13e: per-floor arrival rates (the 2024 paper's rate channels). Order MUST match
            // ObservationSize: 2*F = [from-rate per floor, then to-rate per floor].
            if (obs.arrivalRates)
            {
                int F2 = cfg.numFloors;
                if (_rateScratch == null || _rateScratch.Length != 2 * F2) _rateScratch = new float[2 * F2];
                FillArrivalRates(_rateScratch);
                for (int i = 0; i < 2 * F2; i++) sensor.AddObservation(_rateScratch[i]);
            }

            // Ceiling/ablation only (EXPERIMENT_PLAN.md E5): exact destination histogram for every
            // waiting and in-car rider. No real controller has this before boarding -- it exists
            // purely to measure how much headroom is left if destinations were fully known.
            if (obs.omniscientDestinations)
            {
                for (int f = 0; f < F; f++)
                {
                    WriteDestHistogram(sensor, upQ[f], F, cfg.maxQueue);
                    WriteDestHistogram(sensor, downQ[f], F, cfg.maxQueue);
                }
                for (int i = 0; i < E; i++)
                    WriteDestHistogram(sensor, cars[i].inService ? cars[i].riders : null, F, cfg.capacity);
            }
        }

        static readonly float[] _destHistBuf = new float[64]; // reused scratch buffer, F never exceeds this
        static void WriteDestHistogram(VectorSensor sensor, System.Collections.Generic.List<Passenger> riders, int F, float norm)
        {
            System.Array.Clear(_destHistBuf, 0, F);
            if (riders != null)
                for (int i = 0; i < riders.Count; i++)
                {
                    int d = riders[i].dest;
                    if (d >= 0 && d < F) _destHistBuf[d] += 1f;
                }
            for (int d = 0; d < F; d++) sensor.AddObservation(Mathf.Min(1f, _destHistBuf[d] / norm));
        }

        // ---- Floor-grid observation (EXPERIMENT_PLAN.md E13b: spatial floor-axis conv) ----
        // Emits per-floor local state as a (1 x F x FloorGridFeatures) single-channel "image" so
        // ML-Agents' visual CNN encoder (match3, min-res 5) convolves over the FLOOR axis -- the
        // native analog of the 2024 Traffic-Pattern-Aware paper's Conv1d-over-floors, without any
        // custom torch. The flat MLP treats floors as an unordered concatenation and cannot exploit
        // floor adjacency (a call 1 floor from a car is very different from 10 floors away, but the
        // flat obs makes both look like arbitrary vector indices); the conv gives that inductive
        // bias. Feature order below is arbitrary (the width axis is NOT semantically ordered, a minor
        // impurity vs. a true 1D conv -- the point is the height/floor axis), 8 features so width>=5.
        public const int FloorGridFeatures = 8;

        // Single source of truth for the per-floor grid features, written into a flat buffer in
        // height-major/width-minor order: buf[f * FloorGridFeatures + c]. Both the training-side
        // FloorGridSensor (via WriteFloorGrid) and the eval-side ConvDispatcher call this so the two
        // paths cannot drift.
        public void FillFloorGrid(float[] buf)
        {
            int F = cfg.numFloors, E = cfg.numElevators;
            float invE = E > 0 ? 1f / E : 0f;
            for (int f = 0; f < F; f++)
            {
                int carsHere = 0, carsWant = 0;
                for (int i = 0; i < E; i++)
                {
                    if (!cars[i].inService) continue;
                    if (cars[i].Floor == f) carsHere++;
                    if (cars[i].WantsFloor(f)) carsWant++;
                }
                int o = f * FloorGridFeatures;
                buf[o + 0] = upQ[f].Count > 0 ? 1f : 0f;
                buf[o + 1] = downQ[f].Count > 0 ? 1f : 0f;
                buf[o + 2] = OldestWaitFrac(upQ[f]);
                buf[o + 3] = OldestWaitFrac(downQ[f]);
                buf[o + 4] = Mathf.Min(1f, upQ[f].Count / (float)cfg.maxQueue);
                buf[o + 5] = Mathf.Min(1f, downQ[f].Count / (float)cfg.maxQueue);
                buf[o + 6] = carsHere * invE;
                buf[o + 7] = carsWant * invE;
            }
        }

        // ---- Origin x Destination matrix (EXPERIMENT_PLAN.md E13d) ----
        // A (2 x F x F) tensor: [channel = up/down hall queue] x [height = ORIGIN floor] x
        // [width = DESTINATION floor], value = waiting riders at that origin bound for that dest
        // (normalized by maxQueue, same as the E5 omniscient block).
        //
        // Why this and not the E13b (1 x F x 8) floor grid:
        //   1. BOTH axes are ordered floors, so a 3x3 conv kernel is genuinely meaningful — it sees
        //      "flow from floors near i to floors near j". The E13b grid's width axis was the 8
        //      arbitrary features, which have no spatial ordering: convolving across them imposed a
        //      FALSE inductive bias and wasted capacity.
        //   2. It carries information the flat obs does NOT have (destinations), so it can move the
        //      ASYMPTOTE. E13b only re-presented info already in the flat 254 obs, which is exactly
        //      why its advantage eroded to noise by ~7M steps (faster learning, same endpoint).
        //   3. F x F (16x16 on M) clears the RESNET encoder's min-resolution of 15, unlocking a much
        //      stronger encoder than match3 (the 8-wide E13b grid was capped at match3, min-res 5).
        //   4. It is DCS-realistic: destination-control systems (Schindler/Otis/KONE) genuinely know
        //      HALL destinations before boarding — see the ObservationConfig.omniscientDestinations
        //      tooltip. This is the deployable half of E5's omniscient block, given the structure that
        //      E5 destroyed by flattening it into 592 MLP inputs.
        // Note each channel is triangular (up-queues only ever target dest > origin, down-queues
        // dest < origin); the empty half is harmless and the conv can learn it.
        public const int FloorODChannels = 2;

        // Flat layout matches ObservationWriter's NCHW index (TensorExtensions.Index(n,c,h,w) =
        // c*H*W + h*W + w for n=0), so the eval dispatcher can feed this buffer straight into a
        // (1, 2, F, F) tensor. Single source of truth for train + eval, like FillFloorGrid.
        public void FillFloorOD(float[] buf)
        {
            int F = cfg.numFloors;
            System.Array.Clear(buf, 0, FloorODChannels * F * F);
            float norm = cfg.maxQueue > 0 ? cfg.maxQueue : 1;
            for (int origin = 0; origin < F; origin++)
            {
                AccumulateOD(buf, upQ[origin], 0, origin, F, norm);
                AccumulateOD(buf, downQ[origin], 1, origin, F, norm);
            }
        }

        static void AccumulateOD(float[] buf, System.Collections.Generic.List<Passenger> q,
            int channel, int origin, int F, float norm)
        {
            if (q == null) return;
            int rowBase = channel * F * F + origin * F;
            for (int i = 0; i < q.Count; i++)
            {
                int d = q[i].dest;
                if (d >= 0 && d < F) buf[rowBase + d] += 1f;
            }
            for (int d = 0; d < F; d++)
            {
                int idx = rowBase + d;
                if (buf[idx] > 0f) buf[idx] = Mathf.Min(1f, buf[idx] / norm);
            }
        }

        float[] _odScratch;
        public void WriteFloorOD(ObservationWriter writer)
        {
            int F = cfg.numFloors;
            int need = FloorODChannels * F * F;
            if (_odScratch == null || _odScratch.Length != need) _odScratch = new float[need];
            FillFloorOD(_odScratch);
            for (int c = 0; c < FloorODChannels; c++)
                for (int h = 0; h < F; h++)
                    for (int w = 0; w < F; w++)
                        writer[c, h, w] = _odScratch[c * F * F + h * F + w];
        }

        float[] _gridScratch;
        public void WriteFloorGrid(ObservationWriter writer)
        {
            int F = cfg.numFloors;
            if (_gridScratch == null || _gridScratch.Length != F * FloorGridFeatures)
                _gridScratch = new float[F * FloorGridFeatures];
            FillFloorGrid(_gridScratch);
            for (int f = 0; f < F; f++)
                for (int c = 0; c < FloorGridFeatures; c++)
                    writer[0, f, c] = _gridScratch[f * FloorGridFeatures + c]; // [channel, height=floor, width=feature]
        }

        // ---------------------------------------------------------------- per-car observation (E6)
        // Same ObservationConfig blocks as WriteObservation, but "own car" blocks (carFloor/
        // carActive/carButtons/carMotion/carLoads) cover only carIndex's car instead of all E cars —
        // this is what makes the observation size independent of fleet size, which is the whole
        // point of the multi-agent/parameter-sharing architecture (EXPERIMENT_PLAN.md E6): the same
        // policy network works for any car count since every car agent gets an identically-shaped
        // observation. "Shared" blocks (hall buttons/queues/time/pattern) are building-wide and
        // identical across all car agents, same as before.
        public int CarObservationSize()
        {
            int F = cfg.numFloors, s = 0;
            if (obs.carFloor) s += F;
            if (obs.carActive) s += 1;
            if (obs.carButtons) s += F;
            if (obs.hallButtons) s += 2 * F;
            if (obs.hallCallAge) s += 2 * F;
            if (obs.carMotion) s += 4;
            if (obs.carLoads) s += 1;
            if (obs.queueLengths) s += 2 * F;
            if (obs.timeOfDay) s += 2;
            if (obs.pattern) s += 5;
            return s;
        }

        public void WriteCarObservation(VectorSensor sensor, int carIndex)
        {
            int F = cfg.numFloors;
            var c = cars[carIndex];

            if (obs.carFloor)
            {
                if (c.inService) sensor.AddOneHotObservation(Mathf.Clamp(c.Floor, 0, F - 1), F);
                else for (int k = 0; k < F; k++) sensor.AddObservation(0f);
            }

            if (obs.carActive) sensor.AddObservation(c.inService ? 1f : 0f);

            if (obs.carButtons)
                for (int f = 0; f < F; f++) sensor.AddObservation(c.inService && c.WantsFloor(f) ? 1f : 0f);

            if (obs.hallButtons)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(upQ[f].Count > 0 ? 1f : 0f);
                    sensor.AddObservation(downQ[f].Count > 0 ? 1f : 0f);
                }

            if (obs.hallCallAge)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(OldestWaitFrac(upQ[f]));
                    sensor.AddObservation(OldestWaitFrac(downQ[f]));
                }

            if (obs.carMotion)
            {
                if (c.inService)
                {
                    int m = c.state == CarState.Moving ? (c.dir > 0 ? 2 : 0) : 1;
                    sensor.AddOneHotObservation(m, 3);
                    sensor.AddObservation(F > 1 ? c.position / (F - 1) : 0f);
                }
                else { sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); sensor.AddObservation(0f); }
            }

            if (obs.carLoads) sensor.AddObservation(c.inService ? c.Load : 0f);

            if (obs.queueLengths)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(Mathf.Min(1f, upQ[f].Count / (float)cfg.maxQueue));
                    sensor.AddObservation(Mathf.Min(1f, downQ[f].Count / (float)cfg.maxQueue));
                }

            if (obs.timeOfDay)
            {
                float frac = traffic.DayFraction(simTime);
                sensor.AddObservation(Mathf.Sin(2f * Mathf.PI * frac));
                sensor.AddObservation(Mathf.Cos(2f * Mathf.PI * frac));
            }

            if (obs.pattern)
                sensor.AddOneHotObservation((int)ActivePattern, 5);
        }

        // ------------------------------------------------ split obs for BufferSensor attention (E6-B)
        // Architecture B (EXPERIMENT_PLAN.md E6 / plan Architecture B) feeds the SAME information as
        // the flat WriteObservation, but reorganized: building-wide state goes to a VectorSensor
        // (WriteGlobalObservation) and each car becomes one fixed-size entity in a BufferSensor
        // (WriteCarEntity). ML-Agents then runs a shared per-entity encoder + self-attention over the
        // car entities. "Own-car" blocks (carFloor/carActive/carButtons/carMotion/carLoads) become the
        // per-car entity; "global" blocks (hall buttons/age, queue lengths, time, pattern) are the
        // vector obs. Together these cover exactly the same blocks as WriteObservation.

        public int GlobalObservationSize()
        {
            int F = cfg.numFloors, s = 0;
            if (obs.hallButtons) s += 2 * F;
            if (obs.hallCallAge) s += 2 * F;
            if (obs.queueLengths) s += 2 * F;
            if (obs.timeOfDay) s += 2;
            if (obs.pattern) s += 5;
            return s;
        }

        public void WriteGlobalObservation(VectorSensor sensor)
        {
            int F = cfg.numFloors;

            if (obs.hallButtons)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(upQ[f].Count > 0 ? 1f : 0f);
                    sensor.AddObservation(downQ[f].Count > 0 ? 1f : 0f);
                }

            if (obs.hallCallAge)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(OldestWaitFrac(upQ[f]));
                    sensor.AddObservation(OldestWaitFrac(downQ[f]));
                }

            if (obs.queueLengths)
                for (int f = 0; f < F; f++)
                {
                    sensor.AddObservation(Mathf.Min(1f, upQ[f].Count / (float)cfg.maxQueue));
                    sensor.AddObservation(Mathf.Min(1f, downQ[f].Count / (float)cfg.maxQueue));
                }

            if (obs.timeOfDay)
            {
                float frac = traffic.DayFraction(simTime);
                sensor.AddObservation(Mathf.Sin(2f * Mathf.PI * frac));
                sensor.AddObservation(Mathf.Cos(2f * Mathf.PI * frac));
            }

            if (obs.pattern)
                sensor.AddOneHotObservation((int)ActivePattern, 5);
        }

        public int CarEntitySize()
        {
            int F = cfg.numFloors, s = 0;
            if (obs.carFloor) s += F;
            if (obs.carActive) s += 1;
            if (obs.carButtons) s += F;
            if (obs.carMotion) s += 4;
            if (obs.carLoads) s += 1;
            return s;
        }

        /// <summary>Fills <paramref name="buf"/> (length must == CarEntitySize()) with one car's entity.</summary>
        public void WriteCarEntity(float[] buf, int carIndex)
        {
            int F = cfg.numFloors, o = 0;
            var c = cars[carIndex];

            if (obs.carFloor)
            {
                if (c.inService) buf[o + Mathf.Clamp(c.Floor, 0, F - 1)] = 1f;
                o += F;
            }

            if (obs.carActive) { buf[o] = c.inService ? 1f : 0f; o += 1; }

            if (obs.carButtons)
            {
                for (int f = 0; f < F; f++) buf[o + f] = c.inService && c.WantsFloor(f) ? 1f : 0f;
                o += F;
            }

            if (obs.carMotion)
            {
                if (c.inService)
                {
                    int m = c.state == CarState.Moving ? (c.dir > 0 ? 2 : 0) : 1; // 0 down, 1 stopped, 2 up
                    buf[o + m] = 1f;
                    buf[o + 3] = F > 1 ? c.position / (F - 1) : 0f;
                }
                o += 4;
            }

            if (obs.carLoads) { buf[o] = c.inService ? c.Load : 0f; o += 1; }
        }

        /// <summary>Longest current wait in a hall queue, normalized by maxWait (0 if empty).</summary>
        float OldestWaitFrac(List<Passenger> q)
        {
            if (q.Count == 0) return 0f;
            float maxW = 0f;
            for (int i = 0; i < q.Count; i++) if (q[i].waitTime > maxW) maxW = q[i].waitTime;
            return Mathf.Clamp01(maxW / cfg.maxWait);
        }
    }
}
