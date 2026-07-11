using System.Collections.Generic;

namespace ElevatorRL.Stats
{
    /// <summary>
    /// Subscribes to a Building's passenger events and accumulates the §1 records: an aggregate
    /// EpisodeStats plus per-floor and per-time-bucket breakdowns. Engine-free.
    ///
    /// Integration: construct with the live Building, then call <see cref="Sample"/> once per
    /// decision (it drives occupancy sampling and window rolling), and <see cref="Finish"/> at
    /// episode end to materialize the records. A warmup interval discards the startup transient.
    /// </summary>
    public sealed class StatsCollector
    {
        readonly Building _b;
        readonly float _warmup;
        readonly float _bucket;
        readonly int _floors;
        readonly float _maxWait;

        // global
        readonly WaitHistogram _wait, _ride;
        int _delivered, _abandoned, _rejected, _arrivals;
        int _decisions;

        // per-floor (indexed by origin floor for wait/abandon/origins; dest floor for destinations)
        readonly WaitHistogram[] _fWait;
        readonly int[] _fOrigins, _fDest, _fDelivered, _fAbandoned, _fRejected;
        readonly double[] _fQueueSum;
        readonly float[] _fQueueMax;
        int _floorSamples;

        // fleet occupancy
        double _utilSum; int _utilSamples;
        long _idleCarSamples, _carSamples;
        double _inServiceSum;

        // reward baseline (lifetime totals snapshotted at epoch start)
        double _rw0Total, _rw0Del, _rw0Tow, _rw0Away, _rw0Rej, _rw0Ab, _rw0InEl, _rw0InQ;

        // windows
        readonly List<WindowStats> _windows = new List<WindowStats>();
        WinAcc _cur;
        bool _started;
        float _epochStart, _lastT;

        sealed class WinAcc
        {
            public float start;
            public string pattern;
            public WaitHistogram wait;
            public int delivered, abandoned, rejected;
            public double utilSum, cisSum;
            public int samples;
        }

        public StatsCollector(Building b, float warmupSeconds = 0f, float bucketSeconds = 300f)
        {
            _b = b;
            _warmup = warmupSeconds;
            _bucket = bucketSeconds > 1f ? bucketSeconds : 1f; // guard: 0 would infinite-loop the roll
            _floors = b.cfg.numFloors;
            _maxWait = b.cfg.maxWait;

            _wait = new WaitHistogram(_maxWait);
            _ride = new WaitHistogram(_maxWait * 2f);

            _fWait = new WaitHistogram[_floors];
            for (int f = 0; f < _floors; f++) _fWait[f] = new WaitHistogram(_maxWait);
            _fOrigins = new int[_floors];
            _fDest = new int[_floors];
            _fDelivered = new int[_floors];
            _fAbandoned = new int[_floors];
            _fRejected = new int[_floors];
            _fQueueSum = new double[_floors];
            _fQueueMax = new float[_floors];

            _b.OnArrival += HandleArrival;
            _b.OnDelivered += HandleDelivered;
            _b.OnAbandoned += HandleAbandoned;
            _b.OnRejected += HandleRejected;
        }

        /// <summary>Detach from the Building's events (call when discarding the collector).</summary>
        public void Dispose()
        {
            _b.OnArrival -= HandleArrival;
            _b.OnDelivered -= HandleDelivered;
            _b.OnAbandoned -= HandleAbandoned;
            _b.OnRejected -= HandleRejected;
        }

        bool PastWarmup => _b.simTime >= _warmup;

        // ── event handlers ────────────────────────────────────────────────────
        void HandleArrival(Passenger p)
        {
            if (!PastWarmup) return;
            _arrivals++;
            if (InRange(p.origin)) _fOrigins[p.origin]++;
        }

        void HandleDelivered(Passenger p, float rideSeconds)
        {
            if (!PastWarmup) return;
            _delivered++;
            _wait.Add(p.waitTime);
            _ride.Add(rideSeconds);
            if (InRange(p.origin)) { _fDelivered[p.origin]++; _fWait[p.origin].Add(p.waitTime); }
            if (InRange(p.dest)) _fDest[p.dest]++;
            if (_cur != null) { _cur.delivered++; _cur.wait.Add(p.waitTime); }
        }

        void HandleAbandoned(Passenger p)
        {
            if (!PastWarmup) return;
            _abandoned++;
            if (InRange(p.origin)) _fAbandoned[p.origin]++;
            if (_cur != null) _cur.abandoned++;
        }

        void HandleRejected(int floor)
        {
            if (!PastWarmup) return;
            _rejected++;
            if (InRange(floor)) _fRejected[floor]++;
            if (_cur != null) _cur.rejected++;
        }

        bool InRange(int f) => f >= 0 && f < _floors;

        // ── per-decision sampling ─────────────────────────────────────────────
        /// <summary>Call once per decision. Drives occupancy sampling + window rolling.</summary>
        public void Sample()
        {
            float t = _b.simTime;
            if (t < _warmup) { _lastT = t; return; }

            if (!_started) StartEpoch(t);

            // roll windows up to the current time
            while (t - _cur.start >= _bucket)
                RollWindow(_cur.start + _bucket);

            _decisions++;
            _lastT = t;

            // fleet occupancy: mean load over in-service cars; idle = Idle & empty
            int inService = 0;
            double loadSum = 0;
            for (int i = 0; i < _b.cars.Length; i++)
            {
                var c = _b.cars[i];
                if (!c.inService) continue;
                inService++;
                loadSum += c.Load;
                _carSamples++;
                if (c.state == CarState.Idle && c.riders.Count == 0) _idleCarSamples++;
            }
            float util = inService > 0 ? (float)(loadSum / inService) : 0f;
            _utilSum += util; _utilSamples++;
            _inServiceSum += inService;
            _cur.utilSum += util; _cur.cisSum += inService; _cur.samples++;

            // per-floor queue lengths
            for (int f = 0; f < _floors; f++)
            {
                float q = _b.upQ[f].Count + _b.downQ[f].Count;
                _fQueueSum[f] += q;
                if (q > _fQueueMax[f]) _fQueueMax[f] = q;
            }
            _floorSamples++;
        }

        void StartEpoch(float t)
        {
            _started = true;
            _epochStart = t;
            _rw0Total = _b.RwTotal; _rw0Del = _b.RwDelivered; _rw0Tow = _b.RwToward; _rw0Away = _b.RwAway;
            _rw0Rej = _b.RwRejected; _rw0Ab = _b.RwAbandoned; _rw0InEl = _b.RwInElevator; _rw0InQ = _b.RwInQueue;
            _cur = NewWindow(t);
        }

        WinAcc NewWindow(float start) => new WinAcc
        {
            start = start,
            pattern = _b.ActivePattern.ToString(),
            wait = new WaitHistogram(_maxWait),
        };

        void RollWindow(float boundary)
        {
            FinalizeWindow(_cur);
            _cur = NewWindow(boundary);
        }

        void FinalizeWindow(WinAcc w)
        {
            float dur = _bucket;
            _windows.Add(new WindowStats
            {
                bucketStart = w.start,
                activePattern = w.pattern,
                delivered = w.delivered,
                deliveredRate = dur > 0 ? w.delivered / (dur / 3600f) : 0f,
                waitMean = w.wait.Mean,
                waitP95 = w.wait.P95,
                abandoned = w.abandoned,
                rejected = w.rejected,
                fleetUtilMean = w.samples > 0 ? (float)(w.utilSum / w.samples) : 0f,
                carsInService = w.samples > 0 ? (float)(w.cisSum / w.samples) : 0f,
            });
        }

        // ── materialize ───────────────────────────────────────────────────────
        public EpisodeStats Finish(RunId id)
        {
            if (_started && _cur != null) FinalizeWindow(_cur);

            float measured = _started ? (_lastT - _epochStart) : 0f;
            int demand = _arrivals + _rejected; // rejected never entered a queue
            id.simSeconds = measured;
            id.warmupSeconds = _warmup;

            var e = new EpisodeStats
            {
                id = id,
                delivered = _delivered,
                deliveredPerHour = measured > 0 ? _delivered / (measured / 3600f) : 0f,
                deliveredPerDecision = _decisions > 0 ? _delivered / (float)_decisions : 0f,
                waitMean = _wait.Mean,
                waitP50 = _wait.P50,
                waitP95 = _wait.P95,
                waitMax = _wait.Max,
                rideMean = _ride.Mean,
                rideP95 = _ride.P95,
                abandoned = _abandoned,
                rejected = _rejected,
                abandonRate = demand > 0 ? _abandoned / (float)demand : 0f,
                rejectRate = demand > 0 ? _rejected / (float)demand : 0f,
                waitStd = _wait.Std,
                waitTailRatio = _wait.P50 > 1e-3f ? _wait.P95 / _wait.P50 : 0f,
                rwTotal = (float)(_b.RwTotal - _rw0Total),
                rwDelivered = (float)(_b.RwDelivered - _rw0Del),
                rwToward = (float)(_b.RwToward - _rw0Tow),
                rwAway = (float)(_b.RwAway - _rw0Away),
                rwRejected = (float)(_b.RwRejected - _rw0Rej),
                rwAbandoned = (float)(_b.RwAbandoned - _rw0Ab),
                rwInElevator = (float)(_b.RwInElevator - _rw0InEl),
                rwInQueue = (float)(_b.RwInQueue - _rw0InQ),
                utilFleetMean = _utilSamples > 0 ? (float)(_utilSum / _utilSamples) : 0f,
                idleFraction = _carSamples > 0 ? (float)(_idleCarSamples / (double)_carSamples) : 0f,
                inServiceMean = _utilSamples > 0 ? (float)(_inServiceSum / _utilSamples) : 0f,
            };
            return e;
        }

        public List<FloorStats> BuildFloorStats(RunId id)
        {
            var list = new List<FloorStats>(_floors);
            for (int f = 0; f < _floors; f++)
            {
                list.Add(new FloorStats
                {
                    id = id,
                    floor = f,
                    carsServing = CarsServing(f),
                    origins = _fOrigins[f],
                    destinations = _fDest[f],
                    delivered = _fDelivered[f],
                    abandoned = _fAbandoned[f],
                    rejected = _fRejected[f],
                    waitMean = _fWait[f].Mean,
                    waitP95 = _fWait[f].P95,
                    waitMax = _fWait[f].Max,
                    queueLenMean = _floorSamples > 0 ? (float)(_fQueueSum[f] / _floorSamples) : 0f,
                    queueLenMax = _fQueueMax[f],
                });
            }
            return list;
        }

        public List<WindowStats> BuildWindowStats(RunId id)
        {
            foreach (var w in _windows) w.id = id;
            return _windows;
        }

        /// <summary>Global delivered-wait histogram (post-warmup) — the distribution behind the
        /// p50/p95/max points, for ECDF / histogram export.</summary>
        public WaitHistogram WaitHist => _wait;

        int CarsServing(int floor)
        {
            int n = 0;
            for (int i = 0; i < _b.cars.Length; i++)
            {
                var c = _b.cars[i];
                if (c.inService && floor >= c.minFloor && floor <= c.maxFloor) n++;
            }
            return n;
        }
    }
}
