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

        Acc _acc;
        readonly System.Random _rng;
        bool _patternLoaded;

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
            _patternLoaded = false;

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
            TrafficPattern want = traffic.useDayCycle ? traffic.PatternForTime(simTime) : traffic.defaultPattern;
            if (force || !_patternLoaded || want != ActivePattern)
            {
                ActivePattern = want;
                arrivals.LoadPattern(want);
                _patternLoaded = true;
            }
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
                    if (q.Count >= cfg.maxQueue) { _acc.rejected++; RejectedTotal++; }
                    else q.Add(p);
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
                if (q[i].waitTime >= cfg.maxWait) { q.RemoveAt(i); _acc.abandoned++; AbandonedTotal++; }
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
            float r = reward.delivered * _acc.delivered
                    + reward.movedToward * _acc.toward
                    + reward.movedAway * _acc.away
                    + reward.rejected * _acc.rejected
                    + reward.abandoned * _acc.abandoned
                    + reward.inElevator * _acc.riderSeconds
                    + reward.inQueue * _acc.queueSeconds;
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
            if (obs.carMotion) s += 4 * E;
            if (obs.carLoads) s += E;
            if (obs.queueLengths) s += 2 * F;
            if (obs.timeOfDay) s += 2;
            if (obs.pattern) s += 5;
            return s;
        }

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
        }
    }
}
