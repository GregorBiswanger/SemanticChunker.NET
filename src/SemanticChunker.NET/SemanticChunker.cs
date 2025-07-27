﻿using ICU4N.Text;
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
public sealed class SemanticChunker(
    IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
    int tokenLimit,
    int bufferSize = 1,
    BreakpointThresholdType thresholdType = BreakpointThresholdType.Percentile,
    double? thresholdAmount = null,
    int? targetChunkCount = null,
    int minChunkChars = 0)
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
        IList<string> sentences = SplitIntoSentences(text);

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

        IList<string> contextualSentences = BuildContextualSentences(sentences, bufferSize);

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

    private static IList<string> SplitIntoSentences(string text)
    {
        var result = new List<string>();
        BreakIterator iterator = BreakIterator.GetSentenceInstance();
        iterator.SetText(text);

        for (int start = iterator.First(), end = iterator.Next();
             end != BreakIterator.Done;
             start = end, end = iterator.Next())
        {
            string sentence = text.Substring(start, end - start).Trim();
            if (sentence.Length > 0)
            {
                result.Add(sentence);
            }
        }

        return result;
    }

    private static IList<string> BuildContextualSentences(IList<string> sentences, int buffer)
    {
        var result = new List<string>(sentences.Count);

        for (int i = 0; i < sentences.Count; i++)
        {
            IEnumerable<string> contextBefore = sentences.Skip(Math.Max(0, i - buffer)).Take(buffer);
            IEnumerable<string> contextAfter = sentences.Skip(i + 1).Take(buffer);

            result.Add(string.Join(' ', contextBefore.Concat([sentences[i]]).Concat(contextAfter)));
        }

        return result;
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

            if (chunkText.Length > _maximumChunkCharacters)
            {
                chunkText = chunkText[.._maximumChunkCharacters];
            }

            Embedding<float> embedding = await embeddingGenerator.GenerateAsync(chunkText, cancellationToken: cancellationToken);

            chunks.Add(new Chunk
            {
                Id = Guid.NewGuid().ToString(),
                Text = chunkText,
                Embedding = embedding
            });

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
                Percentile(distances, amount),

            BreakpointThresholdType.StandardDeviation =>
                distances.Average() + amount * StandardDeviation(distances),

            BreakpointThresholdType.InterQuartile =>
                distances.Average() + amount *
                (Percentile(distances, 75) - Percentile(distances, 25)),

            BreakpointThresholdType.Gradient =>
                Percentile(Gradient(distances), amount),

            _ => throw new ArgumentOutOfRangeException(nameof(type))
        };
    }

    private static double ThresholdFromTargetCount(IReadOnlyList<double> distances, int desiredChunks)
    {
        int maxChunks = distances.Count;
        int minChunks = 1;

        int clampedChunks = Math.Clamp(desiredChunks, minChunks, maxChunks);

        double y1 = 0;   // percentile for maxChunks
        double y2 = 100; // percentile for minChunks

        double percentile = maxChunks == minChunks
            ? y2
            : y1 + (y2 - y1) * (clampedChunks - maxChunks) / (minChunks - maxChunks);

        return Percentile(distances, percentile);
    }

    // ---------- Statistics helpers ----------

    private static double Percentile(IReadOnlyList<double> sequence, double p)
    {
        double[] sorted = sequence.OrderBy(v => v).ToArray();
        double n = (sorted.Length - 1) * p / 100d;
        int k = (int)Math.Floor(n);
        double d = n - k;

        return k + 1 < sorted.Length
            ? sorted[k] + d * (sorted[k + 1] - sorted[k])
            : sorted[^1];
    }

    private static double StandardDeviation(IReadOnlyList<double> sequence)
    {
        double average = sequence.Average();
        double variance = sequence.Sum(v => Math.Pow(v - average, 2)) / sequence.Count;
        return Math.Sqrt(variance);
    }

    private static double[] Gradient(IReadOnlyList<double> sequence)
    {
        var g = new double[sequence.Count];

        for (int i = 1; i < sequence.Count - 1; i++)
        {
            g[i] = (sequence[i + 1] - sequence[i - 1]) / 2d;
        }

        g[0] = sequence[1] - sequence[0];
        g[^1] = sequence[^1] - sequence[^2];

        return g;
    }

    private static double CosineSimilarity(IReadOnlyList<float> a, IReadOnlyList<float> b)
    {
        double dot = 0;
        double magnitudeA = 0;
        double magnitudeB = 0;

        for (int i = 0; i < a.Count; i++)
        {
            dot += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        return dot / (Math.Sqrt(magnitudeA) * Math.Sqrt(magnitudeB));
    }
}