using System.Reflection;
using Shouldly;

namespace SemanticChunkerNET.Tests;

public class BuildContextualSentencesTests
{
    private static IList<string> InvokeBuildContextualSentences(IList<string> sentences, int buffer)
    {
        var method = typeof(SemanticChunker).GetMethod(
            "BuildContextualSentences",
            BindingFlags.NonPublic | BindingFlags.Static);

        if (method == null)
        {
            throw new InvalidOperationException("BuildContextualSentences method not found");
        }

        var result = method.Invoke(null, [sentences, buffer]);
        return result as IList<string> ?? throw new InvalidOperationException("Unexpected result type");
    }

    [Fact]
    public void BuildContextualSentences_WithBuffer3_ShouldNotContainDuplicateSentences()
    {
        // Arrange
        List<string> sentences = ["0", "1", "2", "3", "4", "5", "6"];
        int buffer = 3;

        // Act
        var contextual = InvokeBuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(7);
        
        // Expected results based on the fix:
        // i=0: context should be "0 1 2 3" (0 sentences before + sentence 0 + 3 sentences after)
        // i=1: context should be "0 1 2 3 4" (1 sentence before + sentence 1 + 3 sentences after)
        // i=2: context should be "0 1 2 3 4 5" (2 sentences before + sentence 2 + 3 sentences after)
        // i=3: context should be "0 1 2 3 4 5 6" (3 sentences before + sentence 3 + 3 sentences after)
        // i=4: context should be "1 2 3 4 5 6" (3 sentences before + sentence 4 + 2 sentences after)
        // i=5: context should be "2 3 4 5 6" (3 sentences before + sentence 5 + 1 sentence after)
        // i=6: context should be "3 4 5 6" (3 sentences before + sentence 6 + 0 sentences after)
        
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
        var contextual = InvokeBuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(4);
        
        // Expected results:
        // i=0: "A B" (0 before + A + 1 after)
        // i=1: "A B C" (1 before + B + 1 after)
        // i=2: "B C D" (1 before + C + 1 after)
        // i=3: "C D" (1 before + D + 0 after)
        
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
        var contextual = InvokeBuildContextualSentences(sentences, buffer);

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
        var contextual = InvokeBuildContextualSentences(sentences, buffer);

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
        var contextual = InvokeBuildContextualSentences(sentences, buffer);

        // Assert
        contextual.Count.ShouldBe(3);
        
        // Negative buffer should be clamped to 0, so each sentence should only contain itself
        // This ensures the current sentence is always included (no regression from previous implementation)
        contextual[0].ShouldBe("A");
        contextual[1].ShouldBe("B");
        contextual[2].ShouldBe("C");
    }
}
