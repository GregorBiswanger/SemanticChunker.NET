using Shouldly;

namespace SemanticChunkerNET.Tests;

public class BuildContextualSentencesTests
{

    [Fact]
    public void BuildContextualSentences_WithBuffer3_ShouldNotContainDuplicateSentences()
    {
        // Arrange
        List<string> sentences = ["0", "1", "2", "3", "4", "5", "6"];
        int buffer = 3;

        // Act
        var contextual = TextSegmenter.BuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(7);

        contextual[0].ShouldBe("0 1 2 3");
        contextual[1].ShouldBe("0 1 2 3 4");
        contextual[2].ShouldBe("0 1 2 3 4 5");
        contextual[3].ShouldBe("0 1 2 3 4 5 6");
        contextual[4].ShouldBe("1 2 3 4 5 6");
        contextual[5].ShouldBe("2 3 4 5 6");
        contextual[6].ShouldBe("3 4 5 6");
    }

    [Fact]
    public void BuildContextualSentences_WithBuffer1_ShouldHaveCorrectContext()
    {
        // Arrange
        List<string> sentences = ["A", "B", "C", "D"];
        int buffer = 1;

        // Act
        var contextual = TextSegmenter.BuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(4);

        contextual[0].ShouldBe("A B");
        contextual[1].ShouldBe("A B C");
        contextual[2].ShouldBe("B C D");
        contextual[3].ShouldBe("C D");
    }

    [Fact]
    public void BuildContextualSentences_WithBuffer0_ShouldOnlyIncludeCurrentSentence()
    {
        // Arrange
        List<string> sentences = ["X", "Y", "Z"];
        int buffer = 0;

        // Act
        var contextual = TextSegmenter.BuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(3);

        // Each sentence should only contain itself
        contextual[0].ShouldBe("X");
        contextual[1].ShouldBe("Y");
        contextual[2].ShouldBe("Z");
    }

    [Fact]
    public void BuildContextualSentences_WithLargeBuffer_ShouldIncludeAllSentences()
    {
        // Arrange
        List<string> sentences = ["1", "2", "3"];
        int buffer = 10;

        // Act
        var contextual = TextSegmenter.BuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(3);

        // All contexts should contain all sentences since buffer is larger than list
        contextual[0].ShouldBe("1 2 3");
        contextual[1].ShouldBe("1 2 3");
        contextual[2].ShouldBe("1 2 3");
    }

    [Fact]
    public void BuildContextualSentences_WithNegativeBuffer_ShouldIncludeOnlyCurrentSentence()
    {
        // Arrange
        List<string> sentences = ["A", "B", "C"];
        int buffer = -5;

        // Act
        var contextual = TextSegmenter.BuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(3);

        // Negative buffer should be clamped to 0, so each sentence should only contain itself
        // This ensures the current sentence is always included (no regression from previous implementation)
        contextual[0].ShouldBe("A");
        contextual[1].ShouldBe("B");
        contextual[2].ShouldBe("C");
    }
}
