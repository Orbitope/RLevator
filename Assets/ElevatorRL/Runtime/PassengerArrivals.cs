using System;

namespace ElevatorRL
{
    /// <summary>
    /// Generates arrivals: a Poisson count per floor (rate = lambda/sec) and a
    /// per-origin destination distribution (rows sum to 1, diagonal = 0). Pattern
    /// presets regenerate both arrays. All randomness flows through one seedable RNG
    /// so episodes are reproducible.
    /// </summary>
    public sealed class PassengerArrivals
    {
        public readonly int floors;
        public float[] lambda;        // arrivals per second, per floor
        public float[][] destDist;    // destDist[origin][dest]

        readonly Random _rng;

        public PassengerArrivals(int floors, int seed)
        {
            this.floors = floors;
            _rng = new Random(seed);
            lambda = new float[floors];
            destDist = new float[floors][];
            for (int i = 0; i < floors; i++) destDist[i] = new float[floors];
            LoadPattern(TrafficPattern.UpPeak);
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

        public void LoadPattern(TrafficPattern p)
        {
            int NF = floors, top = NF - 1;

            // ---- per-floor arrival rates (arrivals / second) ----
            switch (p)
            {
                case TrafficPattern.UpPeak:
                    Fill(lambda, 0.05f); lambda[0] = 0.9f; break;
                case TrafficPattern.DownPeak:
                    Fill(lambda, 0.22f); lambda[0] = 0.04f; break;
                case TrafficPattern.Lunch:
                    Fill(lambda, 0.10f); lambda[0] = 0.5f; lambda[top] = 0.45f; break;
                case TrafficPattern.Midday:
                    Fill(lambda, 0.15f); break;
                default: // Uniform
                    Fill(lambda, 0.20f); break;
            }

            // ---- destination distribution per origin ----
            for (int o = 0; o < NF; o++)
            {
                float[] row = destDist[o];
                Array.Clear(row, 0, NF);

                switch (p)
                {
                    case TrafficPattern.UpPeak:
                        if (o == 0) { for (int d = 1; d < NF; d++) row[d] = 1f; }
                        else { row[0] = 4f; for (int d = 1; d < NF; d++) if (d != o) row[d] = 0.4f; }
                        break;
                    case TrafficPattern.DownPeak:
                        if (o == 0) { for (int d = 1; d < NF; d++) row[d] = 1f; }
                        else { row[0] = 6f; for (int d = 1; d < NF; d++) if (d != o) row[d] = 0.3f; }
                        break;
                    case TrafficPattern.Lunch:
                        if (o == 0) { row[top] = 4f; for (int d = 1; d < NF; d++) if (d != top) row[d] = 0.4f; }
                        else if (o == top) { row[0] = 4f; for (int d = 1; d < top; d++) row[d] = 0.4f; }
                        else { row[0] = 2f; if (top != o) row[top] = 2f; for (int d = 1; d < NF; d++) if (d != o) row[d] += 0.2f; }
                        break;
                    default: // Midday / Uniform
                        for (int d = 0; d < NF; d++) if (d != o) row[d] = 1f;
                        break;
                }

                Normalize(row);
            }
        }

        static void Fill(float[] a, float v) { for (int i = 0; i < a.Length; i++) a[i] = v; }

        static void Normalize(float[] row)
        {
            float s = 0f;
            for (int i = 0; i < row.Length; i++) s += row[i];
            if (s <= 0f) return;
            for (int i = 0; i < row.Length; i++) row[i] /= s;
        }
    }
}
