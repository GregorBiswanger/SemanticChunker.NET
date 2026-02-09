using Microsoft.Extensions.AI;
using System.Collections.Immutable;

namespace SemanticChunkerNET;

/// <summary>
/// Splits long text into semantically coherent chunks based on sentence embeddings
/// and a configurable breakpoint threshold.
/// </summary>
/// <param name="embeddingGenerator">
///    Service that converts a string into an <see cref="Embedding{TValue}"/>.
/// </param>
/// <param name="tokenLimit">
///    Maximum large language model token budget for a single chunk.
///    A safety margin of ten percent is subtracted automatically.
/// </param>
/// <param name="bufferSize">
///    Number of surrounding sentences taken into account when building the
///    contextual sentence windows from which embeddings are created.
/// </param>
/// <param name="thresholdType">
///    Statistical method that determines the breakpoint threshold.
/// </param>
/// <param name="thresholdAmount">
///    Optional override for the chosen <paramref name="thresholdType"/>.  
///    If omitted, a well‑established default is used.
/// </param>
/// <param name="targetChunkCount">
///    Desired number of chunks.  
///    If specified, the threshold is derived from this value instead of the
///    methods defined by <paramref name="thresholdType"/> and
///    <paramref name="thresholdAmount"/>.
/// </param>
/// <param name="minChunkChars">
///    Chunks shorter than this value are skipped entirely.
/// </param>
/// <param name="maxOverrunChars">
///    When a chunk exceeds the maximum character limit, the splitter searches
///    forward up to this many characters for the next newline boundary before
///    falling back to a hard cut.  Set to 0 to disable boundary‑aware splitting.
/// </param>
public sealed class SemanticChunker(
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    int tokenLimit,
    int bufferSize = 1,
    BreakpointThresholdType thresholdType = BreakpointThresholdType.Percentile,
    double? thresholdAmount = null,
    int? targetChunkCount = null,
    int minChunkChars = 0,
    int maxOverrunChars = 200)
{
    private static readonly IReadOnlyDictionary<BreakpointThresholdType, double> DefaultThresholdAmounts =
        new Dictionary<BreakpointThresholdType, double>
        {
            { BreakpointThresholdType.Percentile,        95 },
            { BreakpointThresholdType.StandardDeviation, 3  },
            { BreakpointThresholdType.InterQuartile,     1.5},
            { BreakpointThresholdType.Gradient,          95 }
        }.ToImmutableDictionary();

    private readonly int _maximumChunkCharacters = (int)(tokenLimit * 4 * 0.9);

    /// <summary>
    /// Creates semantic chunks from the supplied <paramref name="text"/>.
    /// </summary>
    /// <param name="text">Full‑length input text.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>
    /// List of <see cref="Chunk"/> instances that together cover the entire text.
    /// </returns>
    public async Task<IList<Chunk>> CreateChunksAsync(string text, CancellationToken cancellationToken = default)
    {
        IList<string> sentences = TextSegmenter.SplitIntoSentences(text);

        if (sentences.Count <= 1)
        {
            return
            [
                new Chunk
                {
                    Id = Guid.NewGuid().ToString(),
                    Text = text,
                    Embedding = await embeddingGenerator.GenerateAsync(text, cancellationToken: cancellationToken)
                }
            ];
        }

        IList<string> contextualSentences = TextSegmenter.BuildContextualSentences(sentences, bufferSize);

        Embedding<float>[] contextualEmbeddings = await Task.WhenAll(
            contextualSentences.Select(s => embeddingGenerator.GenerateAsync(s, cancellationToken: cancellationToken)));

        double[] distances = CalculateSentenceDistances(contextualEmbeddings);

        double breakpointThreshold = targetChunkCount.HasValue
            ? ThresholdFromTargetCount(distances, targetChunkCount.Value)
            : CalculateThreshold(distances, thresholdType, thresholdAmount ?? DefaultThresholdAmounts[thresholdType]);

        HashSet<int> breakpoints = distances
            .Select((distance, index) => (distance, index))
            .Where(t => t.distance > breakpointThreshold)
            .Select(t => t.index)
            .ToHashSet();

        return await AssembleChunksAsync(sentences, breakpoints, minChunkChars, cancellationToken);
    }

    /// <summary>
    /// Calculates the cosine similarity between two vectors of equal length.
    /// </summary>
    /// <param name="vectorA">The first vector.</param>
    /// <param name="vectorB">The second vector.</param>
    /// <returns>
    /// A value between -1 and1 representing the cosine similarity:
    ///1 indicates identical direction,0 indicates orthogonality, and -1 indicates opposite direction.
    /// Returns0 if either vector has zero magnitude.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown if <paramref name="vectorA"/> or <paramref name="vectorB"/> is null.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown if the vectors do not have the same length.
    /// </exception>
    public static double CosineSimilarity(IReadOnlyList<float> vectorA, IReadOnlyList<float> vectorB)
    {
        if (vectorA is null) throw new ArgumentNullException(nameof(vectorA));
        if (vectorB is null) throw new ArgumentNullException(nameof(vectorB));
        if (vectorA.Count != vectorB.Count) throw new ArgumentException("Vectors must have the same length.");

        double dot = 0, na = 0, nb = 0;

        for (int i = 0; i < vectorA.Count; i++)
        {
            var ai = vectorA[i];
            var bi = vectorB[i];
            dot += ai * bi;
            na += ai * ai;
            nb += bi * bi;
        }

        var denom = Math.Sqrt(na) * Math.Sqrt(nb);
        if (denom == 0) return 0;

        var cos = dot / denom;

        if (cos > 1) return 1;
        if (cos < -1) return -1;

        return cos;
    }



    private static double[] CalculateSentenceDistances(IReadOnlyList<Embedding<float>> embeddings)
    {
        var distances = new double[embeddings.Count - 1];

        for (int i = 0; i < distances.Length; i++)
        {
            float[] vectorA = embeddings[i].Vector.ToArray();
            float[] vectorB = embeddings[i + 1].Vector.ToArray();

            distances[i] = 1d - CosineSimilarity(vectorA, vectorB);
        }

        return distances;
    }

    private async Task<IList<Chunk>> AssembleChunksAsync(
        IList<string> sentences,
        HashSet<int> breakpoints,
        int minimumCharacters,
        CancellationToken cancellationToken)
    {
        var chunks = new List<Chunk>();
        var currentSentences = new List<string>();

        for (int index = 0; index < sentences.Count; index++)
        {
            currentSentences.Add(sentences[index]);

            bool isBreakpoint = breakpoints.Contains(index) || index == sentences.Count - 1;
            if (!isBreakpoint) continue;

            string chunkText = string.Join(' ', currentSentences).Trim();

            if (chunkText.Length < minimumCharacters)
            {
                currentSentences.Clear();
                continue;
            }

            foreach (string part in TextSegmenter.SplitChunkText(chunkText, _maximumChunkCharacters, maxOverrunChars))
            {
                Embedding<float> embedding = await embeddingGenerator.GenerateAsync(part, cancellationToken: cancellationToken);

                chunks.Add(new Chunk
                {
                    Id = Guid.NewGuid().ToString(),
                    Text = part,
                    Embedding = embedding
                });
            }

            currentSentences.Clear();
        }

        return chunks;
    }



    private static double CalculateThreshold(
        IReadOnlyList<double> distances,
        BreakpointThresholdType type,
        double amount)
    {
        return type switch
        {
            BreakpointThresholdType.Percentile =>
                Statistics.Percentile(distances, amount),

            BreakpointThresholdType.StandardDeviation =>
                distances.Average() + amount * Statistics.StandardDeviation(distances),

            BreakpointThresholdType.InterQuartile =>
                distances.Average() + amount *
                (Statistics.Percentile(distances, 75) - Statistics.Percentile(distances, 25)),

            BreakpointThresholdType.Gradient =>
                Statistics.Percentile(Statistics.Gradient(distances), amount),

            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };
    }

    private static double ThresholdFromTargetCount(IReadOnlyList<double> distances, int desiredChunks)
    {
        int maxChunks = distances.Count + 1;
        int minChunks = 1;

        int clampedChunks = Math.Clamp(desiredChunks, minChunks, maxChunks);

        // Special case: if we want max chunks, return a sentinel value below any finite distance
        // so that all distances are above the threshold
        if (clampedChunks == maxChunks)
        {
            return double.NegativeInfinity;
        }

        double y1 = 0;   // percentile for maxChunks
        double y2 = 100; // percentile for minChunks

        double percentile = maxChunks == minChunks
            ? y2
            : y1 + (y2 - y1) * (clampedChunks - maxChunks) / (minChunks - maxChunks);

        return Statistics.Percentile(distances, percentile);
    }


}