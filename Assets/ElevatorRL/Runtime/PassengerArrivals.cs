using System;

namespace ElevatorRL
{
    /// <summary>
    /// Generates arrivals: a Poisson count per floor (rate = lambda/sec) and a per-origin destination
    /// distribution (rows sum to 1, diagonal = 0). All randomness flows through one seedable RNG so
    /// episodes are reproducible.
    ///
    /// EXPERIMENT_PLAN.md E15 — REWRITTEN 2026-07-16 to match the 2024 Traffic-Pattern-Aware paper's
    /// generator (`Passenger.py:PassengerGenerator` + `elevator_controller.py:initialize_arrival_rate`).
    /// The previous model was NOT a harder version of theirs but a different, less realistic problem:
    ///   * ~13.3x more load per unit of fleet carrying capacity (216 vs 16.2 arrivals/hr/slot),
    ///   * a single traffic generator (our "Midday" was 0:100:0 UNIFORM all-to-all — the least
    ///     predictable traffic possible, so a smarter policy had nothing to exploit),
    ///   * one FLAT lambda, constant forever (so arrival-rate observations were near-constant and
    ///     there was no temporal structure to be "traffic-pattern-aware" of).
    /// At that load the optimal policy collapses toward "sweep continuously" — which is what LOOK does
    /// by construction — which plausibly explains the whole "RL never beats LOOK" narrative.
    ///
    /// This model instead uses THREE CONCURRENT components (incoming / interfloor / outgoing), each
    /// driven by a 48 x 15-minute binned day profile (12h) with linear interpolation between bins.
    /// Named patterns are TIME-SLICES of that one continuous day, exactly as the paper does:
    ///   UpPeak = bins 0..11 | InterFloor(Midday) = 12..17 + 30..35 | Lunch = 18..29 | DownPeak = 36..47
    ///   Uniform / useDayCycle = AllInOne, the full 48-bin 12h day.
    /// Resulting mixes match the CIBSE templates researched in E12 (their Lunch is ~23:43:33
    /// incoming:interfloor:outgoing vs CIBSE's 45:45:10 guidance), unlike our old 0:100:0.
    /// </summary>
    public sealed class PassengerArrivals
    {
        public readonly int floors;
        public float[] lambda;        // arrivals per second, per floor
        public float[][] destDist;    // destDist[origin][dest]

        // Live component rates in arrivals/sec (exposed for the E13e arrival-rate observation, which
        // only becomes meaningful now that these actually VARY over time).
        public float rateIn, rateInter, rateOut;

        readonly Random _rng;

        // ---- The paper's arrival-rate profiles, transcribed verbatim from
        // elevator_controller.py:initialize_arrival_rate (the ACTIVE, uncommented tables).
        // 48 bins x 15 min = a 12-hour day. Units are the paper's abstract "ar"; converted to
        // arrivals/sec via RatePerSecond() below using their own inter-arrival formula.
        static readonly float[] ArIn = {
            1.8f, 1.85f, 2.8f, 2.9f, 3.8f, 5.9f, 7.3f, 7f, 7.1f, 4.85f, 3.9f, 3.5f,
            2.1f, 1.6f, 1.4f, 1.7f, 1.6f, 1.55f, 1.5f, 1.45f, 1.6f, 1.5f, 1.65f, 1.7f,
            2.6f, 4.1f, 3.8f, 4.8f, 4f, 2.8f, 1.4f, 2.2f, 2.1f, 1.7f, 1.3f, 1.6f,
            1.5f, 1.4f, 1f, 1.1f, 1f, 1.1f, 0.9f, 0.6f, 0.6f, 0.6f, 0.5f, 0.4f,
        };

        static readonly float[] ArInter = {
            0.3f, 0.6f, 0.6f, 0.55f, 0.7f, 1.2f, 2.2f, 2.5f, 4.5f, 3.2f, 3.4f, 3.95f,
            3.5f, 3.2f, 3f, 4.3f, 3.3f, 2.8f, 4f, 3.25f, 4.2f, 5.5f, 5.3f, 5.4f,
            5.6f, 4.6f, 4.9f, 5.8f, 6f, 3.6f, 1.9f, 1.9f, 2.2f, 2.3f, 1.9f, 2.5f,
            2.6f, 2.4f, 2.3f, 1f, 1.3f, 1.5f, 1.3f, 1f, 0.7f, 0.6f, 0.6f, 0.6f,
        };

        static readonly float[] ArOut = {
            0.4f, 0.4f, 0.25f, 0.55f, 0.4f, 0.3f, 0.6f, 0.7f, 0.7f, 0.8f, 1.3f, 1.3f,
            1.6f, 1.5f, 1.4f, 1f, 1.4f, 1.8f, 4f, 4.8f, 8.2f, 5.7f, 5.8f, 4.2f,
            3.3f, 2.75f, 2.4f, 1.2f, 1.2f, 1.1f, 1.4f, 2.1f, 1.5f, 1.3f, 1.8f, 1.4f,
            1.6f, 2.7f, 2f, 2.5f, 3.4f, 6.4f, 5f, 4.6f, 3.7f, 3.9f, 2.85f, 2f,
        };

        const float BinSeconds = 15f * 60f;
        const int Bins = 48;

        /// <summary>
        /// The paper's inter-arrival formula (`Passenger.py:get_rate`): iat = 300 / (ar * pop / 100),
        /// so arrivals/sec = ar * population / 30000. Their population is 1200 over 20 floors with
        /// 4 cars x capacity 20 (= 80 slots). Ours defaults lower because our fleet is smaller —
        /// see TrafficConfig.loadPerSlot (population is derived per rung in Building.UpdatePattern).
        /// </summary>
        static float RatePerSecond(float ar, int population) => ar * population / 30000f;

        public PassengerArrivals(int floors, int seed)
        {
            this.floors = floors;
            _rng = new Random(seed);
            lambda = new float[floors];
            destDist = new float[floors][];
            for (int i = 0; i < floors; i++) destDist[i] = new float[floors];
            LoadPatternAtTime(TrafficPattern.UpPeak, false, 0f, 600);
        }

        /// <summary>Knuth's Poisson sampler. Returns 0 for non-positive rate.</summary>
        public int Poisson(float l)
        {
            if (l <= 0f) return 0;
            double L = Math.Exp(-l);
            int k = 0;
            double p = 1.0;
            do { k++; p *= _rng.NextDouble(); } while (p > L);
            return k - 1;
        }

        /// <summary>Inverse-CDF sample from the destination row for an origin floor.</summary>
        public int SampleDest(int origin)
        {
            float r = (float)_rng.NextDouble();
            float c = 0f;
            float[] row = destDist[origin];
            for (int d = 0; d < floors; d++)
            {
                c += row[d];
                if (r <= c) return d;
            }
            return origin == 0 ? 1 : 0; // fallback
        }

        /// <summary>Which 15-min bins of the 12h day this pattern occupies (the paper's slices).</summary>
        static int[] BinsFor(TrafficPattern p, bool allInOne)
        {
            if (allInOne) return Range(0, 48);
            switch (p)
            {
                case TrafficPattern.UpPeak:   return Range(0, 12);
                case TrafficPattern.Lunch:    return Range(18, 30);
                case TrafficPattern.DownPeak: return Range(36, 48);
                case TrafficPattern.Midday:   return Concat(Range(12, 18), Range(30, 36)); // paper's InterFloor
                default:                      return Range(0, 48);                          // Uniform = AllInOne
            }
        }

        static int[] Range(int a, int b) { var r = new int[b - a]; for (int i = 0; i < r.Length; i++) r[i] = a + i; return r; }
        static int[] Concat(int[] a, int[] b) { var r = new int[a.Length + b.Length]; a.CopyTo(r, 0); b.CopyTo(r, a.Length); return r; }

        /// <summary>
        /// Interpolate one component's rate within the pattern's bin slice at simTime (the paper's
        /// get_rate: linear between the current bin and the next, wrapping at the end of the slice).
        /// </summary>
        static float InterpAr(float[] table, int[] bins, float simTime)
        {
            float period = bins.Length * BinSeconds;
            float t = simTime % period;
            int idx = (int)(t / BinSeconds);
            if (idx >= bins.Length) idx = bins.Length - 1;
            float shift = t - idx * BinSeconds;
            int next = (idx + 1) % bins.Length;
            float a = table[bins[idx]], b = table[bins[next]];
            return a + shift * (b - a) / BinSeconds;
        }

        /// <summary>
        /// Rebuild lambda[] / destDist[][] for the given pattern at the given sim time. Called every
        /// tick (cheap: 3 interpolations + F + F*F writes) because rates now vary continuously.
        ///
        /// Component -> origin/destination mapping mirrors the paper's get_from_to_floor():
        ///   incoming  : origin = lobby(0),        dest = uniform over upper floors
        ///   outgoing  : origin = uniform upper,   dest = lobby(0)
        ///   interfloor: origin = uniform upper,   dest = uniform other upper (never the lobby)
        /// </summary>
        public void LoadPatternAtTime(TrafficPattern p, bool allInOne, float simTime, int population)
        {
            int NF = floors;
            var bins = BinsFor(p, allInOne);

            rateIn    = RatePerSecond(InterpAr(ArIn,    bins, simTime), population);
            rateInter = RatePerSecond(InterpAr(ArInter, bins, simTime), population);
            rateOut   = RatePerSecond(InterpAr(ArOut,   bins, simTime), population);

            int upper = NF - 1;              // floors 1..NF-1
            if (upper < 1) { Array.Clear(lambda, 0, NF); return; }

            // ---- per-floor origination rates ----
            lambda[0] = rateIn;                                   // all incoming starts at the lobby
            float perUpper = (rateOut + rateInter) / upper;       // outgoing + interfloor spread evenly
            for (int f = 1; f < NF; f++) lambda[f] = perUpper;

            // ---- destination rows ----
            // lobby: incoming goes uniformly to the upper floors
            var row0 = destDist[0];
            Array.Clear(row0, 0, NF);
            for (int d = 1; d < NF; d++) row0[d] = 1f / upper;

            // upper floors: split between "to lobby" (outgoing) and "to another upper" (interfloor)
            float denom = rateOut + rateInter;
            float pLobby = denom > 0f ? rateOut / denom : 1f;
            int others = upper - 1;                              // other upper floors (excl. self)
            for (int o = 1; o < NF; o++)
            {
                var row = destDist[o];
                Array.Clear(row, 0, NF);
                if (others <= 0) { row[0] = 1f; continue; }      // 2-floor building: only the lobby
                row[0] = pLobby;
                float each = (1f - pLobby) / others;
                for (int d = 1; d < NF; d++) if (d != o) row[d] = each;
            }
        }
    }
}
