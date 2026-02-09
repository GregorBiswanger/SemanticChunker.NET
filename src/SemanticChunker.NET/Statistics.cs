namespace SemanticChunkerNET;

/// <summary>
/// Provides statistical helper methods used to determine breakpoint thresholds
/// for semantic chunking.
/// </summary>
internal static class Statistics
{
    /// <summary>
    /// Returns the <paramref name="p"/>-th percentile of <paramref name="sequence"/>
    /// using linear interpolation between the two nearest data points.
    /// </summary>
    internal static double Percentile(IReadOnlyList<double> sequence, double p)
    {
        double[] sorted = sequence.OrderBy(v => v).ToArray();
        double n = (sorted.Length - 1) * p / 100d;
        int k = (int)Math.Floor(n);
        double d = n - k;

        return k + 1 < sorted.Length
            ? sorted[k] + d * (sorted[k + 1] - sorted[k])
            : sorted[^1];
    }

    /// <summary>
    /// Returns the population standard deviation of <paramref name="sequence"/>.
    /// </summary>
    internal static double StandardDeviation(IReadOnlyList<double> sequence)
    {
        double average = sequence.Average();
        double variance = sequence.Sum(v => Math.Pow(v - average, 2)) / sequence.Count;
        return Math.Sqrt(variance);
    }

    /// <summary>
    /// Returns the numerical gradient of <paramref name="sequence"/> using
    /// central differences for interior elements and forward/backward
    /// differences at the boundaries.
    /// </summary>
    internal static double[] Gradient(IReadOnlyList<double> sequence)
    {
        if (sequence.Count == 1)
        {
            return [0.0];
        }

        var g = new double[sequence.Count];

        for (int i = 1; i < sequence.Count - 1; i++)
        {
            g[i] = (sequence[i + 1] - sequence[i - 1]) / 2d;
        }

        g[0] = sequence[1] - sequence[0];
        g[^1] = sequence[^1] - sequence[^2];

        return g;
    }
}
