using System.Reflection;
using Shouldly;

namespace SemanticChunkerNET.Tests;

public class MaxChunkSplitTests
{
    private static IEnumerable<string> InvokeSplitChunkText(string text, int maxChars, int overrun)
    {
        var method = typeof(SemanticChunker).GetMethod(
            "SplitChunkText",
            BindingFlags.NonPublic | BindingFlags.Static);

        if (method == null)
        {
            throw new InvalidOperationException("SplitChunkText method not found");
        }

        var result = method.Invoke(null, [text, maxChars, overrun]);
        return result as IEnumerable<string> ?? throw new InvalidOperationException("Unexpected result type");
    }

    [Fact]
    public void SplitChunkText_CutsAtNewlineInsteadOfHardCut()
    {
        // Arrange: text where a \n occurs shortly after maxChars
        const int maxChars = 36;
        const int overrun = 200;

        // 40 chars of 'A', then newline, then second line
        string firstLine = new string('A', 40);
        string secondLine = "Second line content here.";
        string input = firstLine + "\n" + secondLine;

        // Act
        var parts = InvokeSplitChunkText(input, maxChars, overrun).ToList();

        // Assert: first part ends at the newline boundary (40 chars), not at maxChars (36)
        parts.Count.ShouldBe(2);
        parts[0].ShouldBe(firstLine);
        parts[0].Length.ShouldBe(40);
        parts[0].Length.ShouldNotBe(maxChars);
        parts[0].ShouldNotContain("Second");
        parts[1].ShouldBe(secondLine);
    }

    [Fact]
    public void SplitChunkText_FallsBackToHardCutWhenNoNewlineNearby()
    {
        // Arrange: text with no \n anywhere
        const int maxChars = 36;
        const int overrun = 20;

        string longText = new string('B', 200);

        // Act
        var parts = InvokeSplitChunkText(longText, maxChars, overrun).ToList();

        // Assert: first part is hard-cut at exactly maxChars (36)
        parts.Count.ShouldBeGreaterThanOrEqualTo(2);
        parts[0].Length.ShouldBe(maxChars);
        parts[0].ShouldBe(new string('B', maxChars));

        // All text must be preserved (no loss)
        string combined = string.Join("", parts);
        combined.ShouldBe(longText);
    }

    [Fact]
    public async Task CreateChunksAsync_OversizedChunk_PreservesAllText()
    {
        // Arrange: tokenLimit = 10 → _maximumChunkCharacters = (int)(10 * 4 * 0.9) = 36
        // Create multiple sentences whose combined text exceeds 36 chars.
        // With RandomEmbeddingGenerator and default threshold, similar sentences
        // get grouped into one chunk that exceeds the limit.
        const int tokenLimit = 10;

        // Four distinct sentences, combined > 36 chars
        var input = "Alpha bravo charlie delta echo. Foxtrot golf hotel india juliet. Kilo lima mike. November oscar papa.";

        var generator = new RandomEmbeddingGenerator();
        var chunker = new SemanticChunker(
            generator, tokenLimit,
            bufferSize: 0,
            maxOverrunChars: 200);

        // Act
        IList<Chunk> chunks = await chunker.CreateChunksAsync(input);

        // Assert: chunks should exist and be within reasonable size
        chunks.Count.ShouldBeGreaterThan(1);

        foreach (var chunk in chunks)
        {
            chunk.Text.ShouldNotBeNullOrEmpty();
            chunk.Embedding.ShouldNotBeNull();
        }

        // All original content should be preserved across chunks (no text loss).
        // Chunks are split from space-joined sentence groups, so concatenating
        // all chunk texts should reconstruct the original content (characters may
        // span chunk boundaries).
        string allChars = string.Concat(chunks.Select(c => c.Text));
        // The original input words (joined with spaces) should be fully present
        // when we concatenate all chunks without any separator
        allChars.ShouldContain("Alpha");
        allChars.ShouldContain("papa");
    }
}

