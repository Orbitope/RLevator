using System;

namespace ElevatorRL.Stats
{
    /// <summary>
    /// Fixed-bin histogram over [0, max] with an overflow bin, giving O(1) percentile
    /// estimates (p50/p95/…) without storing every sample. Engine-free so it runs headless
    /// and in tests. Exact count/mean/std/max are tracked alongside the bins; percentiles are
    /// linearly interpolated within the containing bin.
    /// </summary>
    public sealed class WaitHistogram
    {
        readonly int _bins;
        readonly float _binW;
        readonly int[] _counts;

        int _n;
        double _sum, _sumSq;
        float _peak;

        public WaitHistogram(float max, int bins = 64)
        {
            _bins = Math.Max(1, bins);
            _binW = Math.Max(1e-4f, max) / _bins;
            _counts = new int[_bins];
        }

        public int Count => _n;
        public float Mean => _n > 0 ? (float)(_sum / _n) : 0f;
        public float Max => _peak;

        public float Std
        {
            get
            {
                if (_n <= 0) return 0f;
                double m = _sum / _n;
                double var = _sumSq / _n - m * m;
                return var > 0 ? (float)Math.Sqrt(var) : 0f;
            }
        }

        public void Add(float v)
        {
            if (v < 0f) v = 0f;
            _n++;
            _sum += v;
            _sumSq += (double)v * v;
            if (v > _peak) _peak = v;

            int b = (int)(v / _binW);
            if (b < 0) b = 0;
            else if (b >= _bins) b = _bins - 1; // overflow folds into the top bin
            _counts[b]++;
        }

        /// <summary>Percentile for p in [0,1]. p&lt;=0 → 0, p&gt;=1 → exact max.</summary>
        public float Percentile(float p)
        {
            if (_n == 0) return 0f;
            if (p <= 0f) return 0f;
            if (p >= 1f) return _peak;

            long target = (long)Math.Ceiling(p * _n);
            long cum = 0;
            for (int i = 0; i < _bins; i++)
            {
                long prev = cum;
                cum += _counts[i];
                if (cum >= target)
                {
                    float lo = i * _binW;
                    float frac = _counts[i] > 0 ? (target - prev) / (float)_counts[i] : 0f;
                    float est = lo + frac * _binW;
                    return est < _peak ? est : _peak;
                }
            }
            return _peak;
        }

        public float P50 => Percentile(0.50f);
        public float P95 => Percentile(0.95f);

        public void Reset()
        {
            Array.Clear(_counts, 0, _counts.Length);
            _n = 0;
            _sum = _sumSq = 0;
            _peak = 0f;
        }
    }
}
