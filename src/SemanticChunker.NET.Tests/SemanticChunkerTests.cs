using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using OllamaApiFacade.Extensions;
using Shouldly;

namespace SemanticChunker.NET.Tests;

public class SemanticChunkerTests
{
    private readonly Kernel _kernel;
    private const int TokenLimit = 512;

    public SemanticChunkerTests()
    {
        var builder = Kernel.CreateBuilder();

#pragma warning disable SKEXP0010
        builder.Services.AddLmStudioEmbeddingGenerator("text-embedding-multilingual-e5-base");
#pragma warning restore SKEXP0010

        if (System.Diagnostics.Debugger.IsAttached)
        {
            builder.Services.AddProxyForDebug();
        }

        _kernel = builder.Build();
    }

    [Fact]
    public async Task CreateChunksAsync_SingleSentence_ReturnsOneChunkWithEmbedding()
    {
        var input = "Die künstliche Intelligenz verändert derzeit die Art und Weise, wie Unternehmen ihre Prozesse automatisieren und optimieren.";

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chunker = new SemanticChunker(generator, TokenLimit);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        chunks.Count.ShouldBe(1);
        chunks[0].Id.ShouldNotBeNullOrEmpty();
        chunks[0].Text.ShouldBe(input);
        chunks[0].Embedding.ShouldNotBeNull();
        chunks[0].Embedding.Vector.Length.ShouldBeGreaterThan(0);
    }

    [Fact]
    public async Task CreateChunksAsync_MultilineText_ReturnsTwoChunksWithCorrectGrouping()
    {
        var lines = new[]
        {
            "Generative KI eröffnet neue Horizonte.",
            "Sie ermöglicht personalisierte Inhalte in Echtzeit.",
            "Unternehmen profitieren durch gesteigerte Effizienz.",
            "Gleichzeitig entstehen neue ethische Fragestellungen.",
            "Eine verantwortungsvolle Implementierung ist entscheidend."
        };
        var input = string.Join('\n', lines);

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chunker = new SemanticChunker(generator, TokenLimit);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        chunks.Count.ShouldBe(2);

        var expectedFirstChunk = string.Join(' ', lines.Take(2));
        var expectedSecondChunk = string.Join(' ', lines.Skip(2));

        chunks[0].Text.ShouldBe(expectedFirstChunk);
        chunks[1].Text.ShouldBe(expectedSecondChunk);

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }

    [Fact]
    public async Task CreateChunksAsync_DiverseTopics_ReturnsMultipleSemanticallyCohesiveChunks()
    {
        var input = string.Join('\n', new[]
        {
            "Künstliche Intelligenz revolutioniert die moderne Softwareentwicklung.",
            "Machine Learning Algorithmen analysieren große Datenmengen effizient.",
            "Deep Learning erreicht beeindruckende Ergebnisse bei der Mustererkennung.",
            "Klimawandel bedroht Ökosysteme weltweit durch Temperaturanstieg.",
            "Erneuerbare Energien reduzieren CO2-Emissionen nachhaltig.",
            "Naturschutz bewahrt Biodiversität für zukünftige Generationen.",
            "Präventive Medizin verhindert Krankheiten durch Früherkennung.",
            "Telemedizin verbessert Gesundheitsversorgung in ländlichen Gebieten.",
            "Personalisierte Therapien optimieren Behandlungserfolg individuell."
        });

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chunker = new SemanticChunker(generator, TokenLimit);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        chunks.Count.ShouldBeGreaterThan(1);
        chunks.Count.ShouldBeLessThanOrEqualTo(9);

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }

        AnalyzeSemanticCoherence(chunks).ShouldBeGreaterThan(0.0);

        var reconstructed = string.Join(" ", chunks.Select(c => c.Text));
        foreach (var originalLine in input.Split('\n'))
        {
            var firstWord = originalLine.Split(' ').First();
            reconstructed.ShouldContain(firstWord);
        }
    }

    [Fact]
    public async Task CreateChunksAsync_LongMixedText_ReturnsThreeSemanticChunks()
    {
        var lines = new[]
        {
            "Galaxien prallen miteinander und bilden neue Sternhaufen.",
            "Kosmische Hintergrundstrahlung zeugt vom Urknall.",
            "Schwarze Löcher verschlingen Materie mit gigantischer Schwerkraft.",
            "Fachwerkhäuser nutzen seit Jahrhunderten stabile Holzverbindungen.",
            "Brettsperrholz ermöglicht mehrgeschossige Gebäude in kurzer Bauzeit.",
            "Improvisation ist das Herzstück des Jazz.",
            "Synkopierte Rhythmen verleihen der Musik ihre Spannung."
        };
        var input = string.Join('\n', lines);

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chunker = new SemanticChunker(
            generator,
            TokenLimit,
            bufferSize: 0,
            thresholdType: BreakpointThresholdType.Percentile,
            thresholdAmount: 72);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        chunks.Count.ShouldBe(3);
        chunks[0].Text.ShouldBe(string.Join(' ', lines.Take(3)));
        chunks[1].Text.ShouldBe(string.Join(' ', lines.Skip(3).Take(2)));
        chunks[2].Text.ShouldBe(string.Join(' ', lines.Skip(5)));

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }

    [Fact(Skip = "Diagnose‑Test für das Embedding‑Modell")]
    public async Task E5Model_CosineSimilarities_AreWithinExpectedRange()
    {
        var inputs = new[]
        {
            "Hot coffee burns my tongue.",
            "Justice represents moral principles.",
            "armadillo",
            "read",
            "The steaming coffee scalded my mouth."
        };

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var embeddings = new List<Embedding<float>>();

        foreach (var sentence in inputs)
        {
            embeddings.Add(await generator.GenerateAsync(sentence));
        }

        var similarities = new List<double>();
        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = i + 1; j < inputs.Length; j++)
            {
                similarities.Add(CosineSimilarity(embeddings[i].Vector.ToArray(),
                                                  embeddings[j].Vector.ToArray()));
            }
        }

        similarities.ShouldAllBe(x => x >= 0.6 && x <= 1.0);
    }

    [Fact]
    public async Task CreateChunksAsync_TwoSimilarSentences_ReturnsOneOrTwoChunksContainingBothSentences()
    {
        var input = string.Join('\n', new[]
        {
            "Machine learning algorithms analyze data patterns.",
            "Artificial intelligence systems process information automatically."
        });

        var generator = _kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chunker = new SemanticChunker(generator, TokenLimit);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        chunks.Count.ShouldBeInRange(1, 2);

        var combinedText = string.Join(" ", chunks.Select(c => c.Text));
        combinedText.ShouldContain("Machine learning");
        combinedText.ShouldContain("Artificial intelligence");

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }

    private static double AnalyzeSemanticCoherence(IList<Chunk> chunks)
    {
        var maxCoherence = 0.0;

        var technologyTerms = new[] { "intelligenz", "machine", "learning", "deep", "algorithmen" };
        var environmentTerms = new[] { "klimawandel", "energie", "naturschutz", "co2", "biodiversität" };
        var healthTerms = new[] { "medizin", "gesundheit", "therapie", "telemedizin", "behandlung" };

        foreach (var chunk in chunks)
        {
            var lower = chunk.Text.ToLowerInvariant();

            var techCount = technologyTerms.Count(term => lower.Contains(term));
            var envCount = environmentTerms.Count(term => lower.Contains(term));
            var healthCount = healthTerms.Count(term => lower.Contains(term));

            var total = techCount + envCount + healthCount;
            if (total == 0) continue;

            var coherence = Math.Max(techCount, Math.Max(envCount, healthCount)) / (double)total;
            maxCoherence = Math.Max(maxCoherence, coherence);
        }

        return maxCoherence;
    }

    private static double CosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
            throw new ArgumentException("Vektoren müssen gleiche Länge haben");

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (var i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
}
