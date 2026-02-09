using System.Reflection;
using Shouldly;

namespace SemanticChunkerNET.Tests;

public class GradientTests
{
    private static double[] InvokeGradient(IReadOnlyList<double> sequence)
    {
        var method = typeof(SemanticChunker).GetMethod(
            "Gradient",
            BindingFlags.NonPublic | BindingFlags.Static);

        if (method == null)
        {
            throw new InvalidOperationException("Gradient method not found");
        }

        var result = method.Invoke(null, [sequence]);
        return result as double[] ?? throw new InvalidOperationException("Unexpected result type");
    }

    [Fact]
    public void Gradient_WithSingleElement_ReturnsZero()
    {
        // Arrange
        var sequence = new List<double> { 0.5 };

        // Act
        var gradient = InvokeGradient(sequence);

        // Assert
        gradient.Length.ShouldBe(1);
        gradient[0].ShouldBe(0.0);
    }

    [Fact]
    public void Gradient_WithTwoElements_ReturnsCorrectDifference()
    {
        // Arrange
        var sequence = new List<double> { 1.0, 3.0 };

        // Act
        var gradient = InvokeGradient(sequence);

        // Assert
        gradient.Length.ShouldBe(2);
        gradient[0].ShouldBe(2.0);  // 3.0 - 1.0
        gradient[1].ShouldBe(2.0);  // 3.0 - 1.0
    }

    [Fact]
    public void Gradient_WithThreeElements_ReturnsCorrectGradients()
    {
        // Arrange
        var sequence = new List<double> { 1.0, 2.0, 4.0 };

        // Act
        var gradient = InvokeGradient(sequence);

        // Assert
        gradient.Length.ShouldBe(3);
        gradient[0].ShouldBe(1.0);   // 2.0 - 1.0
        gradient[1].ShouldBe(1.5);   // (4.0 - 1.0) / 2.0
        gradient[2].ShouldBe(2.0);   // 4.0 - 2.0
    }

    [Fact]
    public void Gradient_WithMultipleElements_ReturnsCorrectGradients()
    {
        // Arrange
        var sequence = new List<double> { 0.0, 1.0, 3.0, 6.0, 10.0 };

        // Act
        var gradient = InvokeGradient(sequence);

        // Assert
        gradient.Length.ShouldBe(5);
        gradient[0].ShouldBe(1.0);   // 1.0 - 0.0
        gradient[1].ShouldBe(1.5);   // (3.0 - 0.0) / 2.0
        gradient[2].ShouldBe(2.5);   // (6.0 - 1.0) / 2.0
        gradient[3].ShouldBe(3.5);   // (10.0 - 3.0) / 2.0
        gradient[4].ShouldBe(4.0);   // 10.0 - 6.0
    }
}
