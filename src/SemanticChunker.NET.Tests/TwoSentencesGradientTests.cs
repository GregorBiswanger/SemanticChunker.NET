using Microsoft.Extensions.AI;
using Shouldly;

namespace SemanticChunkerNET.Tests;

public class TwoSentencesGradientTests
{
    private const int TokenLimit = 512;

    [Fact]
    public async Task CreateChunksAsync_TwoSentencesWithGradient_ShouldNotThrow()
    {
        // Arrange
        var input = "Die künstliche Intelligenz ist interessant. Wir benutzen sie gerne.";
        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(generator, TokenLimit, thresholdType: BreakpointThresholdType.Gradient);

        // Act
        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        // Assert
        chunks.ShouldNotBeNull();
        chunks.Count.ShouldBeGreaterThanOrEqualTo(1);
        chunks.Count.ShouldBeLessThanOrEqualTo(2);

        var combinedText = string.Join(" ", chunks.Select(c => c.Text));
        combinedText.ShouldContain("künstliche Intelligenz");
        combinedText.ShouldContain("benutzen sie gerne");

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }

    [Fact]
    public async Task CreateChunksAsync_TwoSentencesWithGradientAndCustomThreshold_ShouldNotThrow()
    {
        // Arrange
        var input = "First sentence here. Second sentence here.";
        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(
            generator,
            TokenLimit,
            thresholdType: BreakpointThresholdType.Gradient,
            thresholdAmount: 50);

        // Act
        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        // Assert
        chunks.ShouldNotBeNull();
        chunks.Count.ShouldBeGreaterThanOrEqualTo(1);
        chunks.Count.ShouldBeLessThanOrEqualTo(2);

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }

    [Fact]
    public async Task CreateChunksAsync_TwoSentencesWithGradientAndBuffer_ShouldNotThrow()
    {
        // Arrange
        var input = "Machine learning is powerful. Deep learning is advanced.";
        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(
            generator,
            TokenLimit,
            bufferSize: 1,
            thresholdType: BreakpointThresholdType.Gradient);

        // Act
        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        // Assert
        chunks.ShouldNotBeNull();
        chunks.Count.ShouldBeGreaterThanOrEqualTo(1);
        chunks.Count.ShouldBeLessThanOrEqualTo(2);

        foreach (var chunk in chunks)
        {
            chunk.Id.ShouldNotBeNullOrEmpty();
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
            chunk.Embedding.Vector.Length.ShouldBeGreaterThan(0);
        }
    }
}
