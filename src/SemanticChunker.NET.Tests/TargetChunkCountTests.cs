using Shouldly;

namespace SemanticChunkerNET.Tests;

public class TargetChunkCountTests
{
    private const int TokenLimit = 512;

    [Fact]
    public async Task CreateChunksAsync_ThreeSentences_TargetChunkCountThree_ReturnsThreeChunks()
    {
        var input = string.Join(" ", "Test sentence 1.", "Test sentence 2.", "Test sentence 3.");

        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(generator, TokenLimit, targetChunkCount: 3);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        chunks.Count.ShouldBe(3);
    }

    [Fact]
    public async Task CreateChunksAsync_TwoSentences_TargetChunkCountTwo_ReturnsTwoChunks()
    {
        var input = string.Join(" ", "Test sentence 1.", "Test sentence 2.");

        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(generator, TokenLimit, targetChunkCount: 2);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        chunks.Count.ShouldBe(2);
    }

    [Fact]
    public async Task CreateChunksAsync_FourSentences_TargetChunkCountFour_ReturnsFourChunks()
    {
        var input = string.Join(" ", "Test sentence 1.", "Test sentence 2.", "Test sentence 3.", "Test sentence 4.");

        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(generator, TokenLimit, targetChunkCount: 4);

        IList<Chunk> chunks = await chunker.CreateChunksAsync(input, TestContext.Current.CancellationToken);

        chunks.Count.ShouldBe(4);
    }
}
